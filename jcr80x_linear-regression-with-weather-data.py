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
import pandas as pd

df = pd.read_csv("../input/weather.csv")

#df.info()
#df.Description.unique()

df.columns
df.head()
# Dummy variables for the categorical feature Description

import pandas as pd

df_dummies = pd.get_dummies(df, drop_first=True)

df_dummies.head()
# SHUFFLE ROWS TO REMOVE ORDER EFFECTS

from sklearn.utils import shuffle

df_shuffled = shuffle(df_dummies, random_state=42)
# SPLIT COLUMNS IN FEATURE SET, X, AND WHAT WE WANT TO PREDICT, y= Temperature_c

DV = 'Temperature_c'

X = df_shuffled.drop(DV, axis=1)

y = df_shuffled[DV]
# TEST - TRAIN SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_train.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# USING HUMIDITY AS ONLY FACTOR

model.fit(X_train[['Humidity']], y_train)
# PRINT MODEL EQUATION

intercept = model.intercept_

coefficient = model.coef_

#print('Temperature = {0:0.2f} + ({1:0.2f} x Humidity)'.format(intercept, coefficient[0]))
predictions = model.predict(X_test[['Humidity']])
# CREATE SCATTER OF OBSERVED VS PREDICTIONS AND CALCULATE PEARSON R

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

plt.scatter(y_test, predictions)

plt.xlabel('Y Test (True Values)')

plt.ylabel('Predicted Values')

plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, predictions)[0], 2))

#plt.show()
# RESID PLOT W/ SHAPIRO-WILK TEST FOR NORMALITY

import seaborn as sns

from scipy.stats import shapiro

sns.distplot((y_test - predictions), bins = 50)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Histogram of Residuals (Shapiro W p-value = {0:0.3f})'.

format(shapiro(y_test - predictions)[1]))

#plt.show()
from sklearn import metrics

import numpy as np



metrics_df = pd.DataFrame({'Metric': ['MAE', 'MSE', 'RMSE', 'R-Squared'], 'Value': [metrics.mean_absolute_error(y_test, predictions), metrics.mean_squared_error(y_test, predictions), np.sqrt(metrics.mean_squared_error(y_test, predictions)), metrics.explained_variance_score(y_test, predictions)]}).round(3)



#print(metrics_df)