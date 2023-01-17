import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn import metrics

from sklearn.metrics import r2_score
raw_data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
raw_data.shape
raw_data.head()
raw_data.describe(include = 'all')
raw_data.isnull().sum()
data_cleaned = raw_data.drop(['Serial No.'], axis = 1)
data_cleaned.describe(include = 'all')
sns.distplot(data_cleaned['GRE Score'])
q = data_cleaned['GRE Score'].quantile(0.01)

data_1 = data_cleaned[data_cleaned['GRE Score']>q]

data_1.describe(include = 'all')
q = data_1['TOEFL Score'].quantile(0.01)

data_2 = data_1[data_1['TOEFL Score']>q]

data_2.describe(include = 'all')
data_2.columns.values
q = data_2['University Rating'].quantile(0.01)

data_3 = data_2[data_2['University Rating']>q]

data_3.describe(include = 'all')
q = data_3['SOP'].quantile(0.01)

data_4 = data_3[data_3['SOP']>q]

data_4.describe(include = 'all')
data_4
q = data_4['LOR '].quantile(0.01)

data_5 = data_4[data_4['LOR ']>q]

data_5.describe(include = 'all')
sns.distplot(data_5['CGPA'])
q = data_5['CGPA'].quantile(0.01)

data_6 = data_5[data_5['CGPA']>q]

data_6.describe(include  = 'all')
q = data_6['Chance of Admit '].quantile(0.01)

data_7 = data_6[data_6['Chance of Admit ']>q]

data_7.describe(include = 'all')
data_cleaned = data_7.reset_index(drop = True)
data_cleaned.describe(include = 'all')
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))

ax1.scatter(data_cleaned['GRE Score'], data_cleaned['Chance of Admit '])

ax1.set_title('Chance of Admit and GRE Score')

ax2.scatter(data_cleaned['TOEFL Score'], data_cleaned['Chance of Admit '])

ax2.set_title('Chance of Admit and TOEFL Score')

ax3.scatter(data_cleaned['CGPA'],data_cleaned['Chance of Admit '])

ax3.set_title('Chance of Admit and CGPA')



plt.show()
data_cleaned.describe(include = 'all')
targets = data_cleaned['Chance of Admit ']

inputs = data_cleaned.drop(['Chance of Admit '], axis = 1)
#using sklearn.preprocessing -> StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
#using sklearn.modelselection -> train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 365)
x_train.shape, x_test.shape
# Simple Linear Regression
simple = LinearRegression()

simple.fit(x_train[:,0].reshape((-1,1)), y_train)
plt.scatter(x_train[:,0], y_train)

plt.xlabel('GRE Score', size = 10)

plt.ylabel('Chance of Admit', size = 10)

plt.title('Simple Linear Regression')
y_simp = simple.predict(x_test[:,0].reshape((-1,1)))
plt.scatter(x_test[:,0].reshape((-1,1)), y_test)

plt.plot(x_test[:,0].reshape((-1,1)), y_simp, 'r')

plt.xlabel('GRE Score', size = 10)

plt.ylabel('Chance of Admit', size = 10)

plt.title('Simple Linear Regression')
y_simp
print('MAE:', metrics.mean_absolute_error(y_test, y_simp))

print('MSE:', metrics.mean_squared_error(y_test, y_simp))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_simp)))

r2_score(y_test,y_simp)
simple.intercept_
simple.coef_
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('Targets(y_train)', size = 10)

plt.ylabel('Predictions(y_hat)', size = 10)

plt.show()
## R-Squared

reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel('Targets(y_test)', size = 18)

plt.ylabel('Predictions(y_hat_test)', size = 18)



plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])

df_pf.head()
df_pf['Target'] = np.exp(y_test)

df_pf
y_test= y_test.reset_index(drop=True)
y_test.head()
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '% 2f' % x)

df_pf.sort_values(by = ['Difference'])
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

degree=2

polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())

polyreg.fit(x_train,y_train)
y_pred1=polyreg.predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred1))

print('MSE:', metrics.mean_squared_error(y_test, y_pred1))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

r2_score(y_test,y_pred1)
#Ridge Regression
from sklearn.linear_model import Ridge

ridge=Ridge(alpha=0.5)
ridge.fit(x_train,y_train)
y_pred2=ridge.predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred2))

print('MSE:', metrics.mean_squared_error(y_test, y_pred2))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
r2_score(y_test,y_pred2)
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.5)
lasso.fit(x_train,y_train)
y_pred3=lasso.predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred3))

print('MSE:', metrics.mean_squared_error(y_test, y_pred3))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))

r2_score(y_test,y_pred3)
from sklearn.linear_model import ElasticNet

regr = ElasticNet(random_state=0)

regr.fit(x_train,y_train)
y_pred4=regr.predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred4))

print('MSE:', metrics.mean_squared_error(y_test, y_pred4))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))

r2_score(y_test,y_pred4)