import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores = scores - max(scores)
    loss_i = - scores[y[i]] + np.log(np.sum(np.exp(scores)))
    loss += loss_i
    for j in range(num_classes):
        out = np.exp(scores[j]) /np.sum(np.exp(scores))
        if j == y[i]:
            dW[:, j] += (-1 + out) * X[i]
        else:
            dW[:, j] +=out * X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes =W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = scores - np.max(scores, axis=1).reshape(-1, 1)
  out = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1)
  loss = - np.sum(np.log(out[range(num_train), y]))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
    
  dout = out
  dout[range(num_train), y] += -1
  dW = (X.T).dot(dout)
  dW = dW /num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

