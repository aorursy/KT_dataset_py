# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data from the input file

raw_data = pd.read_csv("/kaggle/input/1.04 Car Data.csv")



raw_data.head()
# Describing the data for all columns

raw_data.describe(include = 'all')
# Dropping the 'Model' column since it is insignificant for the analysis for now

data = raw_data.drop(['Model'], axis = 1)

data.describe(include = 'all')
# Checking null values in the data set

data.isnull().sum()
# Dropping all the rows that has null values

data_no_mv = data.dropna(axis = 0)

data_no_mv.describe()
# Plotting price

sns.distplot(data_no_mv['Price'])
# Removing the 1% of outliers

q = data_no_mv['Price'].quantile(0.99)

data_1 = data_no_mv[data_no_mv['Price'] < q]

display(data_1.describe(include = 'all'))

sns.distplot(data_1['Price'])
# Plotting mileage

sns.distplot(data_no_mv['Mileage'])
# Removing the 1% of outliers

q = data_1['Mileage'].quantile(0.99)

data_2 = data_1[data_1['Mileage'] < q]

display(data_2.describe(include = 'all'))

sns.distplot(data_2['Mileage'])
# Plotting engine volume

sns.distplot(data_no_mv['EngineV'])
# Keeping engine volume less than 6.5 since that is the maximum engine volume (all values above 6.5 are invalid)

data_3 = data_2[data_2['EngineV'] < 6.5]

sns.distplot(data_3['EngineV'])
# Plotting year

sns.distplot(data_no_mv['Year'])
# Removing the outliers

q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year'] > q]

sns.distplot(data_4['Year'])
# Resetting index

data_cleaned = data_4.reset_index(drop = True)

data_cleaned.describe(include = 'all')
# Plotting scatter plots

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])

ax1.set_title('Price and Year')



ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])

ax2.set_title('EngineV and Price')



ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])

ax3.set_title('Mileage and Price')



plt.show()
# Taking log of price to display linearity in relation

log_price = np.log(data_cleaned['Price'])

data_cleaned['Log Price'] = log_price



data_cleaned.head()
# Plotting scatter plots

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Log Price'])

ax1.set_title('Log Price and Year')



ax2.scatter(data_cleaned['EngineV'], data_cleaned['Log Price'])

ax2.set_title('Log Price and EngineV')



ax3.scatter(data_cleaned['Mileage'], data_cleaned['Log Price'])

ax3.set_title('Log Price and Mileage')



plt.show()
# Dropping price column

data_cleaned = data_cleaned.drop(['Price'], axis = 1)

data_cleaned.columns.values
# Importing relevant models

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Creating a small table with features and their variance inflation factor

variables = data_cleaned[['Mileage', 'Year', 'EngineV']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif['Features'] = variables.columns



vif
# Dropping year to avoid multicolinearity

data_no_multicolinearity = data_cleaned.drop(['Year'], axis = 1)
# Creating dummy variables

data_with_dummies = pd.get_dummies(data_no_multicolinearity, drop_first = True)

data_with_dummies.head()
# Displaying all columns

data_with_dummies.columns.values
# Changing column order

cols = ['Log Price', 'Mileage', 'EngineV', 'Brand_BMW',

       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',

       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',

       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',

       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']



data_preprocessed = data_with_dummies[cols]

data_preprocessed.head()
# Assigning targets and inputs

targets = data_preprocessed['Log Price']

inputs = data_preprocessed.drop(['Log Price'], axis = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# Standardizing inputs

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split



# Divinding data into train & test

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 123)
# Building linear regression

reg = LinearRegression()

reg.fit(x_train, y_train)
# Plotting regression on train data

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)

plt.xlabel('Targets [y_train]', size = 18)

plt.ylabel('Predictions [y_hat]', size = 18)

plt.xlim(6, 13)

plt.ylim(6, 13)

plt.show()
# Plotting residual (difference between actual and predicted value)

sns.distplot(y_train - y_hat)

plt.title('Residuals PDF', size = 18)

plt.show()
# Calculating score 

reg.score(x_train, y_train)
# Creating summary table with regression coefficiants and features

reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
# Plotting regression on test data

y_hat_test = reg.predict(x_test)

plt.scatter(y_test, y_hat_test, alpha = 0.2)

plt.xlabel('Targets [y_test]', size = 18)

plt.ylabel('Predictions [y_hat_test]', size = 18)

plt.xlim(6, 13)

plt.ylim(6, 13)

plt.show()
# Creating prediction column

df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Prediction'])

df_pf.head()
# Resetting index

y_test = y_test.reset_index(drop = True)

y_test.head()
# Converting log prices back to normal prices using exponential function

df_pf['Target'] = np.exp(y_test)

df_pf.head()
# Adding residual column

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.abs(df_pf['Residual'] / df_pf['Target'] * 100)

df_pf.head()
df_pf.describe()
# Setting format values for float up to two decimal digits

pd.set_option('display.float_format', lambda x: '%.2f' % x)



# Sort values by difference %

df_pf.sort_values(by = ['Difference%'])