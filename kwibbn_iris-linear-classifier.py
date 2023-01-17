import numpy as np

import pandas as pd
def svm_loss_vectorized(W, X, y, reg):

    """

    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    """

    loss = 0.0

    dW = np.zeros(W.shape)

    num_train=X.shape[0]

    num_classes = W.shape[1]



    scores = X.dot(W)



    correct_class_score = scores[np.arange(num_train),y]



    correct_class_score = np.reshape(np.repeat(correct_class_score,num_classes),(num_train,num_classes))

    margin = scores-correct_class_score+1.0

    margin[np.arange(num_train),y]=0



    loss = (np.sum(margin[margin > 0]))/num_train

    loss+=0.5*reg*np.sum(W*W)



    #gradient

    margin[margin>0]=1

    margin[margin<=0]=0



    row_sum = np.sum(margin, axis=1)

    margin[np.arange(num_train), y] = -row_sum

    dW += np.dot(X.T, margin)



    dW/=num_train

    dW += reg * W



    return loss, dW
def softmax_loss_vectorized(W, X, y, reg):

    """

    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    """

    # Initialize the loss and gradient to zero.

    loss = 0.0

    dW = np.zeros_like(W)

    num_classes = W.shape[1]

    num_train = X.shape[0]



    scores=X.dot(W)

    maxLogC = np.max(scores,axis=1)

    maxLogC=np.reshape(np.repeat(maxLogC,num_classes),scores.shape )

    expScores=np.exp(scores+maxLogC)

    exp_correct_class_score = expScores[np.arange(num_train), y]



    #loss

    loss=-np.log(exp_correct_class_score/np.sum(expScores,axis=1))

    loss=sum(loss)/num_train

    loss+=0.5*reg*np.sum(W*W)



    #gradient

    expScoresSumRow=np.reshape(np.repeat(np.sum(expScores,axis=1),num_classes),expScores.shape )

    graidentMatrix=expScores/ expScoresSumRow



    graidentMatrix[np.arange(num_train),y]-=1

    dW = X.T.dot(graidentMatrix)



    dW/=num_train

    dW+=reg*W



    return loss, dW
class LinearClassifier(object):



    def __init__(self):

        self.W = None



    def train(self, X, y, learning_rate=1e-3, reg=0, num_iters=100,

              batch_size=200, verbose=False):

        """

        Train this linear classifier using stochastic gradient descent.

        Inputs:

        - X: A numpy array of shape (N, D) containing training data; there are N

          training samples each of dimension D.

        - y: A numpy array of shape (N,) containing training labels; y[i] = c

          means that X[i] has label 0 <= c < C for C classes.

        - learning_rate: (float) learning rate for optimization.

        - reg: (float) regularization strength.

        - num_iters: (integer) number of steps to take when optimizing

        - batch_size: (integer) number of training examples to use at each step.

        - verbose: (boolean) If true, print progress during optimization.

        Outputs:

        A list containing the value of the loss function at each training iteration.

        """

        num_train, dim = X.shape

        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

        if self.W is None:

            # lazily initialize W

            self.W = 0.001 * np.random.randn(dim, num_classes)



        # Run stochastic gradient descent to optimize W

        loss_history = []

        for it in range(num_iters):



            id = np.random.choice(np.arange(num_train), batch_size)

            X_batch = X[id]

            y_batch = y[id]



            # evaluate loss and gradient

            loss, grad = self.loss(X_batch, y_batch, reg)

            loss_history.append(loss)



            self.W -= learning_rate * grad



            if verbose and it % 100 == 0:

                print('iteration %d / %d: loss %f' % (it, num_iters, loss))



        return loss_history



    def predict(self, X):

        """

        Use the trained weights of this linear classifier to predict labels for

        data points.

        Inputs:

        - X: A numpy array of shape (N, D) containing training data; there are N

          training samples each of dimension D.

        Returns:

        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional

          array of length N, and each element is an integer giving the predicted

          class.

        """



        y_pred = np.argmax(X.dot(self.W), axis=1)



        return y_pred



    def loss(self, X_batch, y_batch, reg):

        """

        Compute the loss function and its derivative.

        Subclasses will override this.

        Inputs:

        - X_batch: A numpy array of shape (N, D) containing a minibatch of N

          data points; each point has dimension D.

        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        - reg: (float) regularization strength.

        Returns: A tuple containing:

        - loss as a single float

        - gradient with respect to self.W; an array of the same shape as W

        """

        pass
class LinearSVM(LinearClassifier):

  """ A subclass that uses the Multiclass SVM loss function """



  def loss(self, X_batch, y_batch, reg):

    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)





class Softmax(LinearClassifier):

  """ A subclass that uses the Softmax + Cross-entropy loss function """



  def loss(self, X_batch, y_batch, reg):

    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
data = pd.read_csv("/kaggle/input/iris-species-uci/Objects.csv")

target = pd.read_csv("/kaggle/input/iris-species-uci/Target.csv", names=["t"])
X = data.to_numpy()

Y = np.squeeze(target.to_numpy())
model = Softmax()
loss_hist = model.train(X, Y, batch_size=X.shape[0], learning_rate=1e-3, num_iters=10000, verbose=True)
np.sum(model.predict(X) == Y) / Y.shape[0]