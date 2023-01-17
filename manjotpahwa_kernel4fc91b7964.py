import math

import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix





start = time.time()



#MNIST_train_df = pd.read_csv('mnist_train.csv')

MNIST_train_df = pd.read_csv('../input/digit-recognizer/train.csv')

print(MNIST_train_df.shape)



# Choose all rows and after the first column only for the X variable.

X_tr = MNIST_train_df.iloc[:10000, 1:] # iloc ensures X_tr will be a dataframe



# Choose only the first column for the Y variable.

y_tr = MNIST_train_df.iloc[:10000, 0]

#print X_tr.head(5)

#print y_tr.head(5)



X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,test_size=0.2, random_state=30, stratify=y_tr)

print(y_train.shape)

print(y_test.shape)

#print y_train.head(5)

print(np.unique(y_train))



pipeline = Pipeline(steps = [('scaler', StandardScaler()),

    ('SVM', SVC(kernel = 'poly'))])



parameters = {'SVM__C': [0.001, 0.1, 1, 100],

              'SVM__gamma': [10, 1, 0.1, 0.01]}



grid = GridSearchCV(pipeline, param_grid = parameters, cv = 5)



grid.fit(X_train, y_train)

print("score = %3.2f" %(grid.score(X_test, y_test)))



y_pred = grid.predict(X_test)

print("confusion matrix: \n ", confusion_matrix(y_test, y_pred))





X_test_kaggle = pd.read_csv('../input/digit-recognizer/test.csv')

print(X_test_kaggle.shape)

y_pred = grid.predict(X_test_kaggle)



prediction = pd.DataFrame(y_pred, columns=['Label'])

prediction.index += 1

prediction = prediction.to_csv('prediction.csv')