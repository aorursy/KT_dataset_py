import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set()
raw_data = pd.read_csv('../input/1.04. Real-life example.csv')

raw_data.head()
raw_data.describe(include='all')
data = raw_data.drop(['Model'], axis=1)

data.describe(include='all')
data.isnull().sum()
data_no_mv = data.dropna(axis=0)
data_no_mv.describe(include='all')
sns.distplot(data_no_mv['Price'])
q = data_no_mv['Price'].quantile(0.99)

data_1 = data_no_mv[data_no_mv['Price']<q]

data_1.describe(include='all')
sns.distplot(data_1['Price'])
sns.distplot(data_no_mv['Mileage'])
q = data_1['Mileage'].quantile(0.99)

data_2 = data_1[data_1['Mileage']<q]
sns.distplot(data_2['Mileage'])
sns.distplot(data_no_mv['EngineV'])
data_3 = data_2[data_2['EngineV']<6.5]
sns.distplot(data_3['EngineV'])
sns.distplot(data_no_mv['Year'])
q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year']>q]
sns.distplot(data_4['Year'])
data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include='all')
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize =(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])

ax1.set_title('Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])

ax2.set_title('EngineV and Year')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])

ax3.set_title('Mileage and Year')



plt.show()
log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price'] = log_price

data_cleaned
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize =(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])

ax1.set_title('Log Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])

ax2.set_title('EngineV and Log Price')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])

ax3.set_title('Mileage and Log Price')



plt.show()
data_cleaned = data_cleaned.drop(['Price'], axis=1)
data_cleaned.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['Mileage', 'Year', 'EngineV']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values, i)for i in range(variables.shape[1])]

vif['Faeture'] = variables.columns

vif
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
data_with_dummies.head()
data_with_dummies.columns.values
cols = ['log_price','Mileage', 'EngineV',  'Brand_BMW',

       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',

       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',

       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',

       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]

data_preprocessed.head()
targets = data_preprocessed['log_price']

inputs = data_preprocessed.drop(['log_price'],axis=1)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
reg = LinearRegression()

reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('Targets (y_train)', size=18)

plt.ylabel('Predictions (y_hat)', size=18)

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
sns.distplot(y_train - y_hat)

plt.title('Residuals PDF', size=18)
reg.score(x_train, y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
data_cleaned['Brand'].unique()
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)

plt.xlabel('Targets (y_test)', size=18)

plt.ylabel('Predictions (y_hat_test)', size=18)

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])

df_pf.head()
df_pf['Targets'] = np.exp(y_test)

df_pf.head()
y_test = y_test.reset_index(drop=True)
df_pf['Targets'] = np.exp(y_test)

df_pf.head()
df_pf['Residual'] = df_pf['Targets'] - df_pf['Predictions']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Targets']*100)

df_pf
df_pf.describe()
pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_pf.sort_values(by=['Difference%'])