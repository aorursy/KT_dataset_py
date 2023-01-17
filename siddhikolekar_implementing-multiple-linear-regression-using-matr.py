import numpy as np
import pandas as pd
from sklearn import linear_model
from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/amspredictionexcel/amsPredictionSheet1-201009-150447 (2).csv')
df.head()
df.describe()
corr=df.corr()
corr.style.background_gradient()
endog = df['ESE']
exog = sm.add_constant(df[['MSE','Attendance','HRS']])
print(exog)
X=exog.to_numpy()
Y= endog.to_numpy()
s1_xt =np.transpose(X)
print(s1_xt)
s2_mull = np.matmul(s1_xt,X)
print(s2_mull)
s3_inv=np.linalg.inv(s2_mull)
print(s3_inv)
s4_mul=np.matmul(s3_inv,s1_xt)
print(s4_mul)
s5_res=np.matmul(s4_mul,Y)
print(s5_res)
mod = sm.OLS(endog,exog)
results = mod.fit()
print(results.summary())
from sklearn import linear_model
X = df[['MSE','Attendance','HRS']]
y = df['ESE']

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
lm.coef_
lm.intercept_