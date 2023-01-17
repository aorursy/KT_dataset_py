import numpy as np

import matplotlib.pyplot as pt

from sklearn import linear_model

import pandas as pd

import math
original_train_set=pd.read_csv('../input/train.csv')

original_test_set=pd.read_csv('../input/test.csv')

##print(original_train_set.shape)

##print(pd.DataFrame(original_train_set).head(5))
train_set=original_train_set.dropna()

##print(train_set.shape)

test_set=original_test_set.dropna()
X=train_set[['x']].as_matrix()

Y=train_set[['y']].as_matrix()

print('mean of X is',np.mean(X),'\n')

print('median of X is',np.median(X),'\n')



print('mean of Y is',np.mean(Y),'\n')

print('median of Y is',np.median(Y),'\n')



Xtest=test_set[['x']].as_matrix()

Ytest=test_set[['y']].as_matrix()


pt.title('Lets see the linear relationship between x and y of training set')

pt.scatter(X,Y,s=5,c='black',marker='*')

pt.xlabel('training_set_X')

pt.ylabel('training_set_Y')

pt.show()
lm=linear_model.LinearRegression()

lm.fit(X,Y)
print('Coeff of determination:',lm.score(X,Y))

print('correlation is:',math.sqrt(lm.score(X,Y)))
p=lm.predict(X)

pt.title('Scatter between predicted values and actual values in training set')

pt.scatter(Y,p,s=5)

pt.xlabel('actual value')

pt.ylabel('predicted value')

pt.show()
pr=lm.predict(Xtest)

pt.title('plot between actual values and predicted values in the test set')

pt.scatter(Ytest,pr,s=9,c='cyan')

pt.xlabel('test values')

pt.ylabel('predicted values')

pt.show()