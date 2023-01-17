import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math

df = pd.read_csv('../input/attendamcemarkscsv/AttendanceMarksSA.csv')
df.head()
df.describe()
corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')
X = df["MSE"]
y = df["ESE"]

sns.scatterplot(X ,y)
endog = df['ESE']
exog = sm.add_constant(df[['MSE']])
print(exog)
# Fit and summarize OLS model
mod = sm.OLS(endog, exog)
results = mod.fit()
print (results.summary())
def RSE(y_true, y_predicted):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse

rse= RSE(df['ESE'],results.predict())
print(rse)
X1 = df["Attendance"]
y1 = df["ESE"]

sns.scatterplot(X1 ,y1)
endog1 = df['ESE']
exog1 = sm.add_constant(df[['Attendance']])
print(exog)
# Fit and summarize OLS model
mod1 = sm.OLS(endog1, exog1)
results1 = mod1.fit()
print (results1.summary())