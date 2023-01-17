# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
raw_data = pd.read_csv("../input/OLX_Car_Data_CSV.csv", encoding = 'unicode_escape')

raw_data.head()
raw_data.describe(include='all')
raw_data.isnull().sum()
## We won't need model. so let's drop it

raw_data.drop(['Model'], inplace=True, axis=1)

raw_data.head()
data_no_mv = raw_data.dropna(axis=0)

data_no_mv.describe(include='all')
import seaborn as sns

sns.set()

sns.distplot(data_no_mv['Price'])
q = data_no_mv["Price"].quantile(0.99)

data1 = data_no_mv[data_no_mv['Price']<q]

data1.describe(include='all')
sns.distplot(data1['Price'])
sns.distplot(data1['KMs Driven'])
q = data_no_mv["KMs Driven"].quantile(0.96)

data2 = data1[(1<data1['KMs Driven']) & (data1['KMs Driven']<q)]

data2.describe(include='all')
sns.distplot(data1['KMs Driven'])
sns.distplot(data1['Year'])
q = data_no_mv["Year"].quantile(0.01)

data3 = data2[q<data_no_mv['Year']]

data3.describe(include='all')
sns.distplot(data1['Year'])
data_cleaned = data3.reset_index(drop=True)
data_cleaned.describe(include='all')
f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])

ax1.set_title('Price vs Year')

ax2.scatter(data_cleaned['KMs Driven'], data_cleaned['Price'])

ax2.set_title('Price vs KMs Driven')

sns.distplot(data_cleaned['Price'])
log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price'] = log_price
f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(15,3))

ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])

ax1.set_title('log_price vs Year')

ax2.scatter(data_cleaned['KMs Driven'], data_cleaned['log_price'])

ax2.set_title('log_price vs KMs Driven')
data_cleaned.drop(['Price'], axis=1, inplace=True)

data_cleaned.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['KMs Driven', 'Year']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif['features'] = variables.columns
vif
sns.scatterplot(data_cleaned['Year'], data_cleaned['KMs Driven'])
data_cleaned.columns.values
data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)
data_with_dummies.head()
data_with_dummies.columns.values
# Just moving log_price to front

cols = ['log_price', 'KMs Driven', 'Year', 'Brand_BMW', 'Brand_Changan',

       'Brand_Chevrolet', 'Brand_Classic & Antiques', 'Brand_Daewoo',

       'Brand_Daihatsu', 'Brand_FAW', 'Brand_Honda', 'Brand_Hyundai',

       'Brand_KIA', 'Brand_Land Rover', 'Brand_Lexus', 'Brand_Mazda',

       'Brand_Mercedes', 'Brand_Mitsubishi', 'Brand_Nissan',

       'Brand_Other Brands', 'Brand_Porsche', 'Brand_Range Rover',

       'Brand_Subaru', 'Brand_Suzuki', 'Brand_Toyota', 'Condition_Used',

       'Fuel_Diesel', 'Fuel_Hybrid', 'Fuel_LPG', 'Fuel_Petrol',

       'Registered City_Ali Masjid', 'Registered City_Askoley',

       'Registered City_Attock', 'Registered City_Badin',

       'Registered City_Bagh', 'Registered City_Bahawalnagar',

       'Registered City_Bahawalpur', 'Registered City_Bela',

       'Registered City_Bhimber', 'Registered City_Chilas',

       'Registered City_Chiniot', 'Registered City_Chitral',

       'Registered City_Dera Ghazi Khan',

       'Registered City_Dera Ismail Khan', 'Registered City_Faisalabad',

       'Registered City_Gujranwala', 'Registered City_Gujrat',

       'Registered City_Haripur', 'Registered City_Hyderabad',

       'Registered City_Islamabad', 'Registered City_Jhelum',

       'Registered City_Kandhura', 'Registered City_Karachi',

       'Registered City_Karak', 'Registered City_Kasur',

       'Registered City_Khairpur', 'Registered City_Khanewal',

       'Registered City_Khanpur', 'Registered City_Khaplu',

       'Registered City_Khushab', 'Registered City_Kohat',

       'Registered City_Lahore', 'Registered City_Larkana',

       'Registered City_Lasbela', 'Registered City_Mandi Bahauddin',

       'Registered City_Mardan', 'Registered City_Mirpur',

       'Registered City_Multan', 'Registered City_Muzaffargarh',

       'Registered City_Nawabshah', 'Registered City_Nowshera',

       'Registered City_Okara', 'Registered City_Pakpattan',

       'Registered City_Peshawar', 'Registered City_Quetta',

       'Registered City_Rahimyar Khan', 'Registered City_Rawalpindi',

       'Registered City_Sahiwal', 'Registered City_Sargodha',

       'Registered City_Sheikhüpura', 'Registered City_Sialkot',

       'Registered City_Sukkar', 'Registered City_Sukkur',

       'Registered City_Tank', 'Registered City_Vehari',

       'Registered City_Wah', 'Transaction Type_Installment/Leasing']
data_preprocessed = data_with_dummies[cols]

data_preprocessed.head()
targets = data_preprocessed['log_price']

inputs = data_preprocessed.drop(["log_price"], axis=1)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):

        self.scaler = StandardScaler(copy, with_mean, with_std)

        self.columns = columns

        self.mean_ = None

        self.std_ = None

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.std_ = np.std(X[self.columns])

        return self

    

    def transform(self, X, y=None, copy=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
columns_to_scale = ['KMs Driven', 'Year']
scaler = CustomScaler(columns_to_scale)

scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)
inputs_scaled.head(10)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, targets, random_state=42, test_size=.2)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)

plt.xlabel('y_train')

plt.ylabel('y_hat')
sns.distplot(y_train-y_hat)

plt.title('Residual PDF')
reg.score(x_train, y_train)
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Feautures'])

reg_summary['Weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_train, y_hat, alpha=.2)

plt.plot([10,16], [10,16], 'r')

plt.xlabel('y_train')

plt.ylabel('y_hat')

plt.xlim([9, 17])

plt.ylim([9, 17])