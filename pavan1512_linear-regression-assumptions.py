# Import the file

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

from patsy import dmatrices
# Read the data

df = pd.read_csv('/kaggle/input/dataset/data.csv')
df.head()
from matplotlib import pyplot as plt 
# Scatter plot between X1 and Output

plt.scatter(df.X1, df.Output)
# Scatter plot between X2 and Output

plt.scatter(df.X2, df.Output)
# Scatter plot between X3 and Output

plt.scatter(df.X3, df.Output)
# Scatter plot between X4 and Output

plt.scatter(df.X4, df.Output)
df.head(2)
df.columns
# Consider only X variables

df1 = df[['X1', 'X2', 'X3', 'X4']]

df1.head(2)
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF = pd.DataFrame()

VIF['feature'] = df1.columns
VIF['VIF'] = [variance_inflation_factor(df1.values, i) 

                          for i in range(len(df1.columns))] 

VIF
model = 'Output ~  X1 + X2 + X3 + X4'

# in Above expression, Output is Y varialble and remaning are X variables.
y, X = dmatrices(model, df, return_type='dataframe')
y.head(2)
X.head(2)
# Now split the (train/test) data into (70/30)

import numpy as np

split_num = np.random.rand(len(X)) < 0.7
X_train = X[split_num]

y_train = y[split_num]

X_test = X[~split_num]

y_test = y[~split_num]
X_train.shape
# Build the model

import statsmodels.api as sm

LR = sm.OLS(y_train, X_train).fit()
LR.summary()
# Predict 'Y' for the test data

y_test_pred = LR.predict(X_test)
y_test_pred = pd.DataFrame(y_test_pred)

y_test_pred.head()
# Rename the column

y_test_pred = y_test_pred.rename(columns = {0:'Output'})
y_test_pred.head()
# Calculate the residual

Residual = y_test - y_test_pred

Residual
# Residual vs Predicted value

plt.scatter(x = y_test_pred, y = Residual)

plt.xlabel('Predicted values')

plt.ylabel('Residuals')
from statsmodels.compat import lzip

import statsmodels.stats.api as sms
# 'Jarque Bera test' for finding the normality of Residuals

name = ['Jarque-Bera test', 'p-value', 'Skewness', 'Kurtosis']
# Perform 'Jarque-Bera' test

JB_test = sms.jarque_bera(Residual)
# Print

lzip(name, JB_test)
Residual.hist(bins = 40)

plt.show()
# White test 

from statsmodels.stats.diagnostic import het_white
keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
result = het_white(Residual, X_test)
lzip(keys, result)