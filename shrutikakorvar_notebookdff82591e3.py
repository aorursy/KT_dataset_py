# Importing the libraries

import pandas as pd

import numpy as np

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

import math

df=pd.read_csv('../input/simplelinearregression/worldometer_data.csv')

df.head()

df.describe()
corr=df.corr()

corr.style.background_gradient(cmap='coolwarm')
x = df["TotalCases"]

y = df["TotalDeaths"]



sns.scatterplot(x,y)
endog = df["TotalDeaths"]

exog = sm.add_constant(df[['TotalCases']])

print(exog)
# Fit and Summerize OLS model

mod=sm.OLS(endog,exog)

results = mod.fit()

print(results.summary())
def RSE(y_true, y_predicted):

    

    y_true = np.array(y_true)

    y_predicted = np.array(y_predicted)

    RSS = np.sum(np.square(y_true - y_predicted))

    

    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse
rse = RSE(df['TotalDeaths'],results.predict())

print(rse)
x1 = df['Population']

y1 = df['TotalDeaths']



sns.scatterplot(x1, y1)
endog = df['TotalDeaths']

exog = sm.add_constant(df[['Population']])

print(exog)