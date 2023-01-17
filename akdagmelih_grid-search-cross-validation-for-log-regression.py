import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
data = pd.read_csv('../input/Breast_cancer_data.csv')

data.head()
data.info()
data.describe()
data.corr()
data['diagnosis'].value_counts()
y = data.diagnosis.values

x = data.drop('diagnosis', axis=1)

x.head(3)
x = (x-np.min(x))/(np.max(x)-np.min(x))

x.describe()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
print('x_train.shape:', x_train.shape)

print('y_train.shape:', y_train.shape)

print('x_test.shape :', x_test.shape)

print('y_test.shape :', x_test.shape)
from sklearn.linear_model import LogisticRegression



# Creating the model:

lr = LogisticRegression() 



# Training the model with the training datas:

lr.fit(x_train, y_train)



print('Scenario_1 score of the logistic regression: ', lr.score(x_test, y_test))
from sklearn.model_selection import GridSearchCV



grid = {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2']}

# C and penalty are logistic regression regularization parameters

# If C is too small model is underfitted, if C is too big model is overfitted.

# l1 and l2 are regularization loss functions (l1=lasso, l2=ridge)



# Creating the model:

lr = LogisticRegression() 



# Creating GridSearchCV model:

lr_cv = GridSearchCV(lr, grid, cv=10) # Using lr model, grid parameters and cross validation of 10 (10 times of accuracy calculation will be applied) 



# Training the model:

lr_cv.fit(x_train, y_train)



print('best paremeters for logistic regression: ', lr_cv.best_params_)

print('best score for logistic regression after grid search cv:', lr_cv.best_score_)
lr_tuned = LogisticRegression(C=100.0, penalty='l1')



lr_tuned.fit(x_train, y_train)



print('Scenario_2 (tuned) logistic regression score: ', lr_tuned.score(x_test, y_test))