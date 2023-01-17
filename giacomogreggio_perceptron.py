# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from random import seed

import math

from matplotlib import rcParams

#set the plot figure size

rcParams["figure.figsize"] = 10,5



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/sonar-data-set/sonar.all-data.csv")

data.shape

data.head()
#from sklearn.linear_model import Perceptron
class Perceptron(object):

    """

    Perceptron classifier.

    

    Parameters

    ----------

    eta: float

        learning rate, the magnitude of change for our weights during each step through our training data.

    n_iter: int

        number of epochs we’ll allow our learning algorithm to iterate through before ending.

        

    Attributes:

    w_: 1d-array

        weights after fittings.

    error_: list

        number of missclassification in every epoch.

    accuracies_: list

        accuracy (test set) in every epoch.

    """ 

    def __init__(self, eta=0.01, n_iter=200):

        #These two lines set the n_iter and eta arguments to instance variables.

        self.n_iter = n_iter

        self.eta = eta

    

    def fit(self,X,y, chooseWeightVector, X_test,y_test):

        """

        Fit method for training data.

        

        Parameters

        ----------

        X: {array-like}, shape=[n_samples,n_features]

            Training vectors, where 'n_samples' is the number of samples and 

            'n_features' is the number of features.

        y: {array-like}, shape=[n_samples]

            Target values.

            

        Returns

        ---------

        self: object

        """

        

        if chooseWeightVector == 1:

            #Create a random weight vector that is a vector with random values + the bias value (that is +1)

            self.w_ = np.random.rand(1 + X.shape[1])

        else:

            #Create weight vector that is a vector with an n-number of 0’s + the bias value (that is +1)

            self.w_ = np.zeros(1 + X.shape[1])

        self.w_[0] = 1

        #print(self.w_)

        

        self.errors_ = []

        self.accuracies_ = []

        

        for _ in range(self.n_iter):

            #zip: Make an iterator that aggregates elements from each of the iterables.

            for xi, target in zip(X, y):

                # w <- w + α(y — f(x))x or alternately

                # w <- w + α(t - o)x

                # predict is: o = sign(w * x) != t

                o = self.predict(xi)

                update = self.eta * (target - o)

                self.w_[1:] += update * xi

                self.w_[0] += update

            self.calc_error(X_test,y_test)

            

    

    def calc_error(self, X, y):

        """

        Calculate the cost function and the accuracies in every epoch.

        """

        errors = 0

        sumOfAccuracy = 0

        for x_t, y_t in zip(X,y):

            y_pred = self.predict(x_t)

            errors += np.square(y_t-y_pred)

            sumOfAccuracy += 1 if y_pred == y_t else 0

        self.errors_.append(errors/(2*len(X)))

        self.accuracies_.append(sumOfAccuracy/len(X))

    

    def net_input(self,X):

        """

        Calculate the net input.

        """

        # sum(wi * xi)

        # w · x + b

        return np.dot(X, self.w_[1:]) + self.w_[0]

    

    def predict(self, X):

        """

        Return class label after unit step.

        """

        #sign(net)

        return np.where(self.net_input(X) >= 0.0, 1, -1)
from sklearn import preprocessing

data.head()

le = preprocessing.LabelEncoder()

#covert last columns values from type string to type float

for i in range (len(data.columns)-1,len(data.columns)):

    data.iloc[:,i] = le.fit_transform(data.iloc[:,i]).astype(float)

#type(data.iloc[1,len(data.columns)-1])

data.head()

from sklearn.model_selection import train_test_split



X = data.drop(['R'],axis=1)

y = data['R']

y = np.where(y == 1.0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)

print("Total number of examples " + str(len(data.index)))

print("Number of training set examples "+ str(len(X_train.index)))

print("Number of test set examples "+ str(len(X_test.index)))

#try different learning rate

learning_rate = [0.0005,0.005,0.05,0.5]

for i in learning_rate:

    p_null = Perceptron(i)

    p_null.fit(X_train.values,y_train, 0, X_test.values, y_test)

    #p_null.fit(X_train.values,y_train)

    #print(p_null.score(X_test.values,y_test))

    

    #plot cost function

    plt.plot(range(1, len(p_null.errors_) + 1), p_null.errors_, marker='o')

    plt.title('learning_rate = {}'.format(i))

    plt.xlabel('Epochs')

    plt.ylabel('Cost function')

    #plt.savefig('images/02_07.png', dpi=300)

    plt.show()

    

    #plot accuracy in every epoch

    plt.plot(range(1, len(p_null.accuracies_) + 1), p_null.accuracies_)

    plt.title('learning_rate = {}'.format(i))

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    #plt.savefig('images/02_07.png', dpi=300)

    plt.show()

    

    print('Mean accuracy on test set with {} epochs and learning rate={}: {} '.format(p_null.n_iter,i,sum(p_null.accuracies_)/(len(p_null.accuracies_))))
seed(1)

for i in learning_rate:

    p_rand = Perceptron(i)

    p_rand.fit(X_train.values,y_train, 1, X_test.values, y_test)

    #p_rand.fit(X_train.values,y_train)

    #p_rand.score(X_test.values,y_test)

    

    #plot cost function

    plt.plot(range(1, len(p_rand.errors_) + 1), p_rand.errors_, marker='o')

    plt.title('learning_rate = {}'.format(i))

    plt.xlabel('Epochs')

    plt.ylabel('Cost function')

    #plt.savefig('images/02_07.png', dpi=300)

    plt.show()

    

    #plot accuracy in every epoch

    plt.plot(range(1, len(p_rand.accuracies_) + 1), p_rand.accuracies_)

    plt.title('learning_rate = {}'.format(i))

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    #plt.savefig('images/02_07.png', dpi=300)

    plt.show()

    

    print('Mean accuracy on test set with {} epochs and learning rate={}: {} '.format(p_rand.n_iter,i,sum(p_rand.accuracies_)/(len(p_rand.accuracies_))))