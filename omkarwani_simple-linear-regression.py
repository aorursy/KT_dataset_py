import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

import math
df = pd.read_csv('../input/attendancemarks/AttendanceMarksSA.csv')

df.head()
df.describe()
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
sns.scatterplot(x=df['MSE'],y=df["ESE"])
endog = df["ESE"]

exog = sm.add_constant(df[["MSE"]])

print(exog)
# Fit and summerize OLS model

mod = sm.OLS(endog,exog)

results = mod.fit()

print(results.summary())
def RSE(y_true,y_predict):

    

    y_true= np.array(y_true)

    y_predict= np.array(y_predict)

    RSS = np.sum(np.square(y_true-y_predict))

    

    RSE = math.sqrt(RSS/(len(y_true)-2))

    return RSE
rse = RSE(df['ESE'],results.predict())

print(rse)
sns.scatterplot(x=df['Attendance'],y=df["ESE"])
endog = df["ESE"]

exog = sm.add_constant(df[['Attendance']])

print(exog)
# Fit and summerrize OLS model

mod = sm.OLS(endog,exog)

results = mod.fit()

print(results.summary())
rse = RSE(df['ESE'],results.predict())

print(rse)