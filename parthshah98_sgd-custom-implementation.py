import numpy as np

import pandas as pd

from sklearn.datasets import make_classification

from math import log

import math

from sklearn.metrics import log_loss
X, y = make_classification(n_samples=50000, n_features=15, n_informative=10, n_redundant=5,

                           n_classes=2, weights=[0.7], class_sep=0.7, random_state=15)
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn import linear_model
# alpha : float

# Constant that multiplies the regularization term. 



# eta0 : double

# The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules.



clf = linear_model.SGDClassifier(eta0=0.0001, alpha=0.0001, loss='log', random_state=15, penalty='l2', tol=1e-3, verbose=2, learning_rate='constant')

clf
clf.fit(X=X_train, y=y_train)
def sigmoid(w,x,b):

  Z = np.dot(w,x)+b

  return(1/(1 + np.exp(-Z)))
def mod(w):

  sum1 = 0

  for i in w:

    sum1  = sum1 + i*i

  return(math.sqrt(sum1)) 
w = np.zeros_like(X_train[0])

b = 0

eta0  = 0.0001

alpha = 0.0001

N = len(X_train)
TRAIN_LOSS = []

TEST_LOSS = []





#Train loss

loss = 0

for i in range(len(X_train)):

  loss += (np.log(sigmoid(w,X_train[i],b)) * y_train[i] + np.log(1 - sigmoid(w,X_train[i],b)) * (1 - y_train[i]))

loss = (-1)*loss/len(X_train) 

TRAIN_LOSS.append(loss)



#Test Loss

loss = 0

for i in range(len(X_test)):

  loss += (np.log(sigmoid(w,X_test[i],b)) * y_test[i] + np.log(1 - sigmoid(w,X_test[i],b)) * (1 - y_test[i]))

loss = (-1)*loss/len(X_test) 

TEST_LOSS.append(loss)
from sklearn.metrics import log_loss

l = len(X_train)

for ep in range(20):



  '''Update weights and intercept for each point'''

  for i in range(len(X_train)):

    w = (1 - (0.0001)/l)*w + 0.0001 * X_train[i] * (y_train[i] - sigmoid(w,X_train[i],b))

    

    b = b + 0.0001*(y_train[i]-sigmoid(w,X_train[i],b))



  ''' Find Loss'''

  #Train loss

  loss = 0

  for i in range(len(X_train)):

    loss += (np.log(sigmoid(w,X_train[i],b)) * y_train[i] + np.log(1 - sigmoid(w,X_train[i],b)) * (1 - y_train[i]))

  loss = (-1)*loss/len(X_train) 

  TRAIN_LOSS.append(loss)



  #Test Loss

  loss = 0

  for i in range(len(X_test)):

    loss += (np.log(sigmoid(w,X_test[i],b)) * y_test[i] + np.log(1 - sigmoid(w,X_test[i],b)) * (1 - y_test[i]))

  loss = (-1)*loss/len(X_test) 

  TEST_LOSS.append(loss)
print("Weights are :",w)



print("Intercept :" ,b)
import matplotlib.pyplot as plt

plt.scatter([i for i in range(21)] ,TRAIN_LOSS, alpha=0.5)

plt.scatter([i for i in range(21)] ,TEST_LOSS, alpha=0.5)

plt.title('LOSS changes as per epoches')

plt.xlabel('epoches')

plt.ylabel('LOSS')

plt.show()
print(TRAIN_LOSS)

print(TEST_LOSS)
w-clf.coef_, b-clf.intercept_
def pred(w,b, X):

    N = len(X)

    predict = []

    for i in range(N):

        if sigmoid(w, X[i], b) >= 0.5: # sigmoid(w,x,b) returns 1/(1+exp(-(dot(x,w)+b)))

            predict.append(1)

        else:

            predict.append(0)

    return np.array(predict)

print(1-np.sum(y_train - pred(w,b,X_train))/len(X_train))

print(1-np.sum(y_test  - pred(w,b,X_test))/len(X_test))