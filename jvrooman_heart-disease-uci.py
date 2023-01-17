# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import needed libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

#import the data

data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.info()
data.columns
sns.heatmap(data.corr(),cmap='RdYlGn', square=True)
#split data between test and train

x = data.drop('target', axis=1)

y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 17)

print(x_train.shape)



print(y_train.shape)
print(x_train.head(3))

print(y_train.head())
logreg = LogisticRegression()

grid = {"C":np.logspace(-5,5,12), "penalty":["l1","l2"]}

logreg_cv = GridSearchCV(logreg, grid, cv = 10)

logreg_cv.fit(x_test, y_test)



print('The best parameter set is: {}'.format(logreg_cv.best_params_))

print('The score is : {}'.format(logreg_cv.best_score_))
y_pred = logreg_cv.predict(x_test)

r2 = logreg_cv.score(x_test, y_test)

mse = mean_squared_error(y_test, y_pred)
print('The r2 score is: {}'.format(r2))

print('The mean squared error is: {}'.format(mse))
logreg2 = logreg_cv.best_estimator_
print('The logistic coefficents are: {}'.format(list(zip(x_test.columns, logreg2.coef_.tolist()[0]))))

print('The logistic intercept is: {}'.format(logreg2.intercept_))