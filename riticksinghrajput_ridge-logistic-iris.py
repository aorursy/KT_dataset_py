

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris_data=load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

iris['target'] = iris_data.target
iris
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
x = iris_data.data 

y = iris_data.target 



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.4, 

                                                    random_state = 0) 

   

print("xtrain shape : ", xtrain.shape) 

print("xtest shape  : ", xtest.shape) 

print("ytrain shape : ", ytrain.shape) 

print("ytest shape  : ", ytest.shape)
lr =  LogisticRegression()

lr.fit(xtrain, ytrain)

y_pred = lr.predict(xtest)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix

print('Accuracy score: ', format(accuracy_score(ytest, y_pred)))

print('Precision score: ', format(precision_score(ytest, y_pred, average='micro')))

print('Recall score: ', format(recall_score(ytest, y_pred, average='micro')))

print('F1 score: ', format(f1_score(ytest, y_pred, average='micro')))

print('\nConfusion Matrix :\n', confusion_matrix(ytest, y_pred))
from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0)

clf.fit(xtrain, ytrain)

y_pred = lr.predict(xtest)


print('Accuracy score: ', format(accuracy_score(ytest, y_pred)))

print('Precision score: ', format(precision_score(ytest, y_pred, average='micro')))

print('Recall score: ', format(recall_score(ytest, y_pred, average='micro')))

print('F1 score: ', format(f1_score(ytest, y_pred, average='micro')))

print('\nConfusion Matrix :\n', confusion_matrix(ytest, y_pred))