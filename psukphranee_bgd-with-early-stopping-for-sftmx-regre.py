# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Define the softmax function.



def softmax(logit):

    #the input, logit, is some N x K matrix. each column containing s_k for each instance

    exps = np.exp(logit)

    #normalize the values of exps. Set keepdims = True so that exp_sums is a column. It turns into a row by default which isn't what we want since we're condensing horizontally

    exp_sums = np.sum(exps, axis=1, keepdims=True)

    return exps / exp_sums

    
#Y_proba, Y_train_one_hot are N by K matrices defined later. N is the number of instances and K number of categories

#loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon)))
from sklearn import datasets

iris = datasets.load_iris()



iris.keys()



#iris['data'] is a 150 by 4 numpy ndarray. Columns 3 and 4 contain the Petal Length and Petal Width, respectively. iris['target'] contains numbers 0-2 indicating the flower type.

import numpy as np



X = iris['data'][:,(2,3)]

y = iris['target']



#Add bias term to X

X_with_bias = np.c_[np.ones([len(X), 1]) ,X]
#Shuffle and split the data and targets into training, testing, and validation sets. 

test_ratio = 0.2

validation_ratio = 0.2

total_size = len(X_with_bias)



#create an array of random numbers for index shuffling

index_shuffle = np.random.permutation(total_size)



test_size = int(test_ratio * total_size)

validation_size = int(validation_ratio * total_size)

train_size = total_size - test_size - validation_size



X_train = X_with_bias[index_shuffle[:train_size]]

y_train = y[index_shuffle[:train_size]]

X_valid = X_with_bias[index_shuffle[train_size:-test_size]] #second index count backwards when slicing

y_valid = y[index_shuffle[train_size:-test_size]] #second index count backwards when slicing

X_test = X_with_bias[index_shuffle[-test_size:]]

y_test = y[index_shuffle[-test_size:]]
def to_one_hot(input):

    #columns

    cols = input.max() + 1

    #rows

    rows = len(input)

    Y_one_hot = np.zeros((rows, cols))

    Y_one_hot[np.arange(rows), input] = 1

    return Y_one_hot
y_train_one_hot = to_one_hot(y_train)

y_test_one_hot = to_one_hot(y_test)

y_valid_one_hot = to_one_hot(y_valid)
n_inputs = X_train.shape[1]

n_outputs = len(np.unique(y_train))



theta_matrix = np.random.randn(n_inputs, n_outputs)
eta = 0.01 #step size

epsilon = 1e-7

n_iterations = 5001

m = len(X_train) # equal to 90



for iteration in range(n_iterations):

    #get the logits by the matrix of instaces by the theta matrix

    logits = X_train.dot(theta_matrix) # 90 by 3 matrix of type ndarray

    

    #Y_proba is a 90 by 3 matrix of normalized exponentials returned by the softmax function

    y_proba = softmax(logits) 

    

    #calculate the loss that was introduced at the beginning of the notebook

    loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis=1))

    

    #calculate the gradient vector. 

    #error are the terms in the parentheses introduced above

    error = y_proba - y_train_one_hot

    

    if iteration % 500 == 0:

        print(iteration, loss)

        

    gradient = 1/m * X_train.T.dot(error)

    

    theta_matrix = theta_matrix - eta * gradient

    