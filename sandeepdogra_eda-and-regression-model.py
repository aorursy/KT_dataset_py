# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for regression plotting
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY 50.csv")
df.head()
df.describe()
df.info()
df.isnull().values.any()
df.isnull().values.all()
#df.isnull().sum() - this will return the count of NULLs/NaN values in each column.
df.isnull().sum() 
#df[df['Volume'] == np.NaN]
df['Turnover'] 
#filling na values with mean for better data presentation
#df.fillna()
mean_volume = df['Volume'].mean()
df["Volume"] = df["Volume"].transform(lambda x: x.fillna(mean_volume))

mean_TO = df['Turnover'].mean()
df["Turnover"] = df["Turnover"].transform(lambda x: x.fillna(mean_TO))
df["Turnover"]
df.tail()
df.corr()
df[['P/E','P/B' ,'Div Yield']].corr()
from matplotlib import pyplot as plt

sns.regplot(x="P/E", y="Div Yield", data=df)
plt.ylim(0,)

sns.regplot(x="P/B", y="Div Yield", data=df)
plt.ylim(0,)

from scipy import stats

pearson_coef, p_value = stats.pearsonr(df['P/E'], df['Div Yield'])
print("The Pearson Correlation Coefficient for P/E w.r. t Div Yiedl is", pearson_coef, " with a P-value of P =", p_value) 

#moderate high negative corelation of P/E with Divident Yied
pearson_coef, p_value = stats.pearsonr(df['P/B'], df['Div Yield'])
print("The Pearson Correlation Coefficient  for P/B w.r. t Div Yiedl is", pearson_coef, " with a P-value of P =", p_value) 

#moderate negative corelation of P/B with Divident Yied
#linear Regression for P/E variable
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

print(lm)
X = df[['P/E']]
Y = df['Div Yield']
lm.fit(X,Y)
Yhat=lm.predict(X)
print(Yhat[0:5])
print(lm.intercept_)
print(lm.coef_)
#linear Regression for P/E variable
lm1 = LinearRegression()

print(lm1)
X = df[['P/B']]
Y = df['Div Yield']
lm1.fit(X,Y)
Yhat=lm.predict(X)
print(Yhat[0:5])
print(lm1.intercept_)
print(lm1.coef_)
#Regression Plot
%matplotlib inline 
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="P/E", y="Div Yield", data=df)
plt.ylim(0,)
plt.figure(figsize=(width, height))
sns.regplot(x="P/B", y="Div Yield", data=df)
plt.ylim(0,)
#Residual Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['P/E'], df['Div Yield'])
plt.show()
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['P/B'], df['Div Yield'])
plt.show()
Y_hat = lm.predict(Z)
Z = df[['P/B', 'P/E']]
lm.fit(Z, df['Div Yield'])
print(lm.intercept_)
print(lm.coef_)
Y_hat = lm.predict(Z)
print(Y_hat)

#Plotting actual values vs fitted values from predicted Y_hat function

plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['Div Yield'], hist=False, color="r", label="Actual Value")

sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')

plt.xlabel('Dividend Yield')
plt.ylabel('Proportion of Dividend Yield')

plt.show()
plt.close()
