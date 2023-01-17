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

# for data visualization
import matplotlib.pyplot as plt
from pylab import rcParams

# ML algorithms and scraping
from scipy import stats
from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Lasso #this is a ML algorithm for linear regression, using feature selection

# Load data
trainingData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
competitionData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# set plotting parameters
%matplotlib inline
rcParams['figure.figsize'] = 10,4
print("Training Data Shape (Rows, Columns):", trainingData.shape)
print("Competition Data Shape (Rows, Columns):", competitionData.shape)
display(trainingData.head())
display(competitionData.head())
# to see all columns
pd.options.display.max_columns = None
display(trainingData.head())
# for discrete items, nan for "missing" and 0 for continuous

trainingDataColumns = list(trainingData)

for c in trainingDataColumns:
    if trainingData[c].dtype == object:
        trainingData[c].fillna(value='Missing', inplace=True)
    else:
        trainingData[c].fillna(0, inplace=True)
        
# do the same for the competition data
competitionDataColumns = list(competitionData)

for f in competitionDataColumns:
    if competitionData[f].dtype == object:
        competitionData[f].fillna(value= 'Missing', inplace=True)
    else:
        competitionData[f].fillna(0, inplace=True)
        
display(trainingData.head())
display(competitionData.head())
# transform discrete values to columns with 1 and 0s
# this dummies transform each discrete column to many columns with value 1 for belongging
trainingData = pd.get_dummies(trainingData)
competitionData = pd.get_dummies(competitionData)

display(trainingData.head())
display(competitionData.head())
print("Traning Data Shape (rows, cols): ", trainingData.shape)
print("Competition Data Shape (rows, cols): ", competitionData.shape)

#display(trainingData.columns.values)
sp = trainingData['SalePrice']
# the value must be the same,
# this inconsistancey occured since maybe there are some discrete values in the training dataset which aren't in the competition
# So, we will try to drop the excessive features that are not presented in both sets


missingFeatures = list(set(trainingData.columns.values) - set(competitionData.columns.values))
trainingData = trainingData.drop(missingFeatures, axis=1)

missingFeatures = list(set(competitionData.columns.values) - set(trainingData.columns.values))
competitionData = competitionData.drop(missingFeatures, axis=1)

print("Traning Data Shape (rows, cols): ", trainingData.shape)
print("Competition Data Shape (rows, cols): ", competitionData.shape)
X_train, X_test, y_train, y_test = train_test_split(trainingData, sp, random_state=0)
# Lasso is a form of linear regression that restricts coefficients to be close to zero or exactly zero.
# this acts as a form of automatic features selection
# alpha is how strongly the coefficients are pushed to zero
# I performed a loop on alpha to get the one that returned the highest test scores, removed for faster performance

myModel = Lasso(alpha=298.4).fit(X_train, y_train)
print("Train score: ", myModel.score(X_train, y_train), "\nTest Score: ", myModel.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(myModel.coef_ !=0)))
submission = pd.DataFrame(myModel.predict(competitionData), columns=['SalePrice'], index = competitionData.index)
display(submission.head())
submission.to_csv("submission_bhart.csv")
