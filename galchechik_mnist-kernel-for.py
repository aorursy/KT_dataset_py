#My first public kernel learning about scikit-learn classification using the MNIST dataset - following tutorial 'Hands-on Machine Learning with Scikit-Learn & Tensorflow' by Aurelien Geron



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
mnist_train = pd.read_csv('../input/train.csv')

print(mnist_train.info())

mnist_test = pd.read_csv('../input/test.csv')

print(mnist_test.info())
print(mnist_train.head())
X_train = np.array(mnist_train)[0:4999, 1:785]

y_train = np.array(mnist_train)[0:4999:, 0]

X_valid = np.array(mnist_train)[5000:5999, 1:785]

y_valid = np.array(mnist_train)[5000:5999, 0]



print(X_train.shape)

print(y_train.shape)

print(X_valid.shape)

print(y_valid.shape)



# We could hav used mnist_test, but the labels are unknown

# X_test = np.array(mnist_test)[:, :] 

#Visualise a digit

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



i = 3798

print(y_train[i])

digit = X_train[i]

digit_image = digit.reshape(28, 28)



plt.imshow(digit_image, cmap= matplotlib.cm.binary, interpolation="nearest")

plt.axis("off")

plt.show()
#Shuffle training set

shuffle_index = np.random.permutation(len(X_train))

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
class NearestNeighbor(object):

  def __init__(self):

    pass



  def train(self, X, y):

    """ X is N x D where each row is an example. Y is 1-dimension of size N """

    # the nearest neighbor classifier simply remembers all the training data

    self.Xtr = X

    self.ytr = y



  def predict(self, X):

    """ X is N x D where each row is an example we wish to predict label for """

    num_test = X.shape[0]

    # lets make sure that the output type matches the input type

    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)



    # loop over all test rows

    for i in range(num_test):

      # find the nearest training image to the i'th test image

      # using the L1 distance (sum of absolute value differences)

      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)

      min_index = np.argmin(distances) # get the index with smallest distance

      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example



    return Ypred
nn = NearestNeighbor() # create a Nearest Neighbor classifier class

nn.train(X_train, y_train) # train the classifier on the training images and labels

y_predict = nn.predict(X_valid) # predict labels on the test images

# and now print the classification accuracy, which is the average number

# of examples that are correctly predicted (i.e. label matches)

print( np.mean(y_predict == y_valid) )
