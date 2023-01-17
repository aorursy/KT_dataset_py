# import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor   # data model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split #data spliter

from learntools.core import *

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer
# read data

# train data path

train_data_path = '../input/train.csv'

test_data_path = '../input/test.csv'



# read train data

train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)
# let see the first few rows of data 

train_data.head()
# let see the first few rows of data 

test_data.head()
#removing ID and sales price form  the features"

train_ID = train_data['Id']

test_ID = test_data['Id']



y = train_data['SalePrice']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train_data.drop("Id", axis = 1, inplace = True)

test_data.drop("Id", axis = 1, inplace = True)
null_columns=train_data.columns[train_data.isnull().any()]

f, ax = plt.subplots(figsize=(20,5))

plt.xticks(rotation='vertical')

plt.bar(null_columns, train_data[null_columns].isnull().sum())
missing_data = train_data.isnull().sum()

missing_data = missing_data.drop((missing_data[missing_data == 0].index)).sort_values(ascending=False)

col_null_value = pd.DataFrame({'Missing Ratio' :missing_data})

col_null_value['% value'] = (col_null_value['Missing Ratio'] / len(train_data))*100

col_null_value
# As We can see the Alley, PoolQC, MiscFeature, 'Fence', 'FireplaceQu', 'LotFrontage' contain most of null values so we will drop them. 

train_data = train_data.drop(columns=['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage' ], axis=1) 

test_data = test_data.drop(columns=['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage' ], axis=1) 
correlation = train_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(correlation, vmax=0.9, square=True)
# delete high correlated column 

train_data = train_data.drop(columns=['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea', '1stFlrSF' ], axis=1) 

test_data = test_data.drop(columns=['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea', '1stFlrSF' ], axis=1) 
corre_sal = train_data.corr()['SalePrice']

correlation['SalePrice'].sort_values(ascending=False)
# selecting cols which are correlated to price

selected_num_col = []

dorp_col_list = []

for col in correlation: 

    if (correlation['SalePrice'][col]) > 0.1 or (correlation['SalePrice'][col]) < -0.1 :

        selected_num_col.append(col)

    else: 

        dorp_col_list.append(col)

        

dorp_col_list
#removing columns with low correlation with sales price

# delete high correlated column 

train_data = train_data.drop(columns=dorp_col_list, axis=1) 

test_data = test_data.drop(columns=dorp_col_list, axis=1) 
sale_price = train_data['SalePrice']

train_data.drop("SalePrice", axis = 1, inplace = True)
#  lets Select numerical columns Again

numerical_cols = [cname for cname in train_data.columns if 

                train_data[cname].dtype in ['int64', 'float64']]

numerical_cols
# some of numerical columns should be categorical column lets convert them

train_data['YearBuilt'] = train_data['YearBuilt'].astype(str)

train_data['YearRemodAdd'] = train_data['YearRemodAdd'].astype(str)



# some of numerical columns should be categorical column lets convert them

test_data['YearBuilt'] = test_data['YearBuilt'].astype(str)

test_data['YearRemodAdd'] = test_data['YearRemodAdd'].astype(str)
#  lets Select numerical columns Again

numerical_cols = [cname for cname in train_data.columns if 

                train_data[cname].dtype in ['int64', 'float64']]

numerical_cols
# plotting all the numerical features against the sales price

f, ax = plt.subplots(figsize=(20,20))

cpt  = 0

for col in numerical_cols: 

    cpt  = cpt + 1

    ax = plt.subplot(4, 5, cpt)

    plt.plot(sale_price, train_data[col], 'o')

    plt.xlabel('Sale Price')

    plt.ylabel(col)
null_num_columns=train_data[numerical_cols].columns[train_data[numerical_cols].isnull().any()]

null_num_columns
#let plot distibution data for the MasVnrArea columna

f, ax = plt.subplots(figsize=(7,5))

sns.distplot(train_data['MasVnrArea'].dropna())
#As we can see the max of the values are zero for MasVsArea so we will fill it with zero

train_data.MasVnrArea = train_data.MasVnrArea.fillna(0)
#select data

X = train_data



X.head()
X_train = pd.get_dummies(train_data)

X_test = pd.get_dummies(test_data)
X, test_X = X_train.align(X_test, join='left', axis=1)
# lets fill the empty values in test data set with mean value

test_X.head()
test_null_col = [test_X.columns[test_X.isnull().any()]]

test_null_col
test_X.head()
#SimpleImputer
test_data_imputed_values = test_X.copy()

train_imputer = X.copy()



imp = Imputer(missing_values='NaN', strategy='mean')

imp.fit(train_imputer)

test_data_imputed_values= pd.DataFrame(imp.transform(test_data_imputed_values))

test_data_imputed_values.columns = test_X.columns
# split test and train data 

x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
xbr_model = XGBRegressor(n_estimators=500, learning_rate=0.06, random_state=5) 

xbr_model.fit(x_train, y_train)

xbr_preds = xbr_model.predict(x_val)

xbr_mea = mean_absolute_error(y_val, xbr_preds)

xbr_mea
mean_value = []

mean_value.append({ "Algorithm" : 'XGBRegressor', "Mean" : xbr_mea })
lin_model = LinearRegression()

lin_model.fit(x_train, y_train)

lin_pred = lin_model.predict(x_val)

lin_mean = mean_absolute_error(y_val, lin_pred)

lin_mean
mean_value.append({ "Algorithm" : 'LinearRegression', "Mean" : lin_mean })


gbr_model = GradientBoostingRegressor(n_estimators=1250, learning_rate=0.04) 

gbr_model.fit(x_train, y_train)

gbr_pred = gbr_model.predict(x_val)

gbr_mean = mean_absolute_error(y_val, gbr_pred)

gbr_mean
mean_value.append({ "Algorithm" : 'GradientBoostingRegressor', "Mean" : gbr_mean })
mean_df = pd.DataFrame(mean_value)

mean_df.set_index('Algorithm', inplace=True)

pd.options.display.float_format = '{:,.2f}'.format

mean_df
test_preds= xbr_model.predict(test_data_imputed_values)



output = pd.DataFrame({'Id': test_ID,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)