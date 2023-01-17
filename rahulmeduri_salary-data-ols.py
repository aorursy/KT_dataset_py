import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()
sns.boxplot(data['Salary'])
print(data['YearsExperience'].skew())

data['YearsExperience'].plot(kind='kde')
print(data['Salary'].skew())

data['Salary'].plot(kind='kde')
# Check for Correlation

data.corr()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
import statsmodels.api as sm
sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()
sns.scatterplot(data['YearsExperience'], data['Salary'])
data.describe()
coeff_of_var = data.std()/data.mean()

coeff_of_var
x = data['YearsExperience']

y = data['Salary']
b1 = np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)

b1
b0 = y.mean()-b1*x.mean()

b0
ypred = b1*x+b0
residue = y-ypred
sse = np.sum((y-ypred)**2)

sse
mse = np.mean((y-ypred)**2)

mse
rmse = np.sqrt(mse)

rmse
sst = np.sum((y.mean()-y)**2)

sst
ssr = np.sum((y.mean()-ypred)**2)

ssr
sse = np.sum((y-ypred)**2)

sse
ssr/sst
plt.plot(x,y,'*')

plt.plot(x,x*b1+b0)

plt.axhline(y.mean(), color='r')
x_c = sm.add_constant(x)
ols_model = sm.OLS(y,x_c).fit()
ols_model.summary()
data_to_predict = pd.DataFrame({'const':1, 'YearsExperience':[3.8,2,4.5,9,15]})

predicted_results = ols_model.predict(data_to_predict)

predicted_results