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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Import Data

winequality_df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
winequality_df.head()
# Checking for NULL values in the dataset

winequality_df[winequality_df.isnull() == True].count()
# Creat a correlation matrix to understand how the independent variables are related to the quality



corr = winequality_df.corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

from sklearn import metrics
regmodel = LinearRegression()
winequality_df.columns
X = winequality_df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

Y = winequality_df[['density']].values
linreg = cross_val_score(estimator=regmodel,  X= X, y= Y, cv=5)
print(linreg)
linpred = cross_val_predict(estimator=regmodel, X= X, y= Y, cv=5)
print('MSE = ', metrics.mean_squared_error(Y,linpred))
# Polynomial Regression

polymodel = PolynomialFeatures(degree= 4)
y = polymodel.fit_transform(Y)
polypred = cross_val_predict(estimator=regmodel, X= X, y= y, cv=5)
polyreg = cross_val_score(estimator=regmodel,  X= X, y= y, cv=5)
print(polyreg)