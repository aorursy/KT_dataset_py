# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


electricity  = pd.read_excel('../input/Folds5x2_pp.xlsx')

print(electricity.info())
electricity.head(3)
electricity.shape
train_sizes = [1, 100, 500, 2000, 5000, 7654]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

features = ['AT', 'V', 'AP', 'RH']
target = 'PE'

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = LinearRegression(), X = electricity[features],
    y = electricity[target], train_sizes=train_sizes, cv=5,
    scoring = 'neg_mean_squared_error')
print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)
train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index=train_sizes))
print('\n', '-'* 20)
print('\nMean validation scores\n\n', pd.Series(validation_scores_mean, index=train_sizes))
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, validation_scores_mean, label='Validation error')

plt.ylabel('MSE', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curve for a linear regression model', fontsize=18, y=1.03)
plt.legend()
plt.ylim(0,40)
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')

def learning_curves(estimator, data, features, target, train_sizes, cv ):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, data[features], data[target], train_sizes = train_sizes,
        cv = cv, scoring = 'neg_mean_squared_error')
    
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error ')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0,40)
    
from sklearn.ensemble import RandomForestRegressor
plt.figure(figsize = (16,5))
for model, i in [(RandomForestRegressor(), 1), (LinearRegression(), 2)]:
    plt.subplot(1,2,i)
    learning_curves(model, electricity, features, target, train_sizes, 5 )
learning_curves(RandomForestRegressor(max_leaf_nodes=350), electricity, features, target, train_sizes, 5)