# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Get the highlights of the data

data = pd.read_csv('../input/Admission_Predict.csv')

data.describe()
data.rename(columns={'LOR ':'LOR'}, inplace = True)
sns.boxplot(x="University Rating", y="GRE Score",

            data=data)

sns.despine(offset=10, trim=True)
sns.boxplot(x="University Rating", y="TOEFL Score",

            data=data)

sns.despine(offset=10, trim=True)
sns.boxplot(x="University Rating", y="SOP",

            data=data)

sns.despine(offset=10, trim=True)
sns.boxplot(x="University Rating", y="LOR",

            data=data)

sns.despine(offset=10, trim=True)
sns.boxplot(x="University Rating", y="CGPA",

            data=data)

sns.despine(offset=10, trim=True)
data.columns
from sklearn.preprocessing import StandardScaler

data2 = data[['GRE Score', 'TOEFL Score', 'SOP',

       'LOR', 'CGPA', 'Research','University Rating']]

scaler = StandardScaler()

data3 = scaler.fit_transform(data2)

data4 = pd.DataFrame(data3)
data4.iloc[6]
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

X = data[['GRE Score', 'TOEFL Score', 'SOP',

       'LOR', 'CGPA', 'Research', 'Chance of Admit ']]

Y = data['University Rating']

X_train = pd.DataFrame(X[:100])

Y_train = pd.DataFrame(Y[:100])

regressor.fit(X_train, Y_train)

X_test = pd.DataFrame(X[100:])

Y_test = pd.DataFrame(Y[100:])

Y_pred = regressor.predict(X_test)

Y_pred_df = pd.DataFrame(Y_pred)

print(regressor.predict(np.array([328, 112, 4, 4.5, 9.1, 1, 0.78]).reshape(1, -1)))
import statsmodels.formula.api as sm

arr = (pd.DataFrame(np.ones(((400,1)), dtype = int)))

X1 = pd.concat([arr, X], axis = 1)

X_opt = X1.iloc[:, [0,1,2,3,4,5,6,7]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

SUMM = regressor_OLS.summary()

regressor_OLS.summary()
regressor_OLS.pvalues.round(3)

def backwardElimination(x, y, SL):

    numVars = len(x.iloc[1, :])

    cols = pd.Series(x.columns)

#    print(cols)

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(endog = y, exog = x).fit()

        pvals = regressor_OLS.pvalues.round(5)

#        print(pvals)

        if max(pvals) >= SL:

            for j in range(0, len(pvals)):

#                print(j)

#                print(pvals[j])

#                print(SL)

#                print(cols.iloc[j])

                if pvals[j] >= SL:

#                    print(cols.iloc[j])

                    x = x.drop(columns = cols.iloc[j])

                    cols = cols.drop(index = j)

                    cols = cols.reset_index(drop=True)

#                    print(x.columns)

#                    print(cols)

                    numVars = numVars - 1

#                    print(numVars)

                    break

        else:

            break

    regressor_OLS = sm.OLS(endog = y, exog = x).fit()

    return x, regressor_OLS



X_best_fit, best_reg_model = backwardElimination(X_opt, Y, 0.05)

best_reg_model.summary()
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

X = X_best_fit

Y = data['University Rating']

X_train = pd.DataFrame(X[:100])

Y_train = pd.DataFrame(Y[:100])

regressor.fit(X_train, Y_train)

X_test = pd.DataFrame(X[100:])

Y_test = pd.DataFrame(Y[100:])

Y_pred = regressor.predict(X_test)

Y_pred_df = pd.DataFrame(Y_pred)

print(regressor.predict(np.array([1, 112, 4, 4.5, 9.1]).reshape(1, -1)))