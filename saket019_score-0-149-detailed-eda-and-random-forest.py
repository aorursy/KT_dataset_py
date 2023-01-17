import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.info()
#dropping Id as it is an irrelevant variable

data.drop(['Id'], inplace=True, axis=1)
data.describe()
#checking skewness in independent variable.

data['SalePrice'].hist(bins=50)
#Improving skewness of the variable using log

data['SalePrice'] = np.log1p(data['SalePrice'])

data['SalePrice'].hist(bins=50)
data.corr()['SalePrice'].sort_values()
plt.figure(figsize = (16, 10))

corr =  data.corr()

kot = corr[corr>=0.7]

sns.heatmap(kot, annot = True, cmap="YlGnBu")

# below ylim adjustment was made as the new matplotlib version cuts off the top and bottom edges by 0.5 value

b, t = plt.ylim()

b += 0.5

t -= 0.5

plt.ylim(b, t)

plt.show()
data.drop(['GarageYrBlt', 'GarageArea', 'TotRmsAbvGrd','MoSold'], inplace=True, axis=1)

# GarageYrBlt and YearBuilt have correlation of 0.83. But YearBuilt has higher correlation with target variable

# GarageArea and GarageCars have correlation of 0.88. But GarageCars has higher correlation with target variable

# TotRmsAbvGrd and GrLivArea have correlation of 0.83. But GrLivArea has higher correlation with target variable

# MoSold seems to be another irrelevant variable. 
data.plot(kind='scatter', x='KitchenAbvGr', y='SalePrice', color='r') 
sum(data['GarageCond'].eq(data['GarageQual']))

# sum(data['OverallCond'].eq(data['OverallQual']))
data.MSSubClass = data.MSSubClass.apply(lambda x: str(x)) # converting it to a categorical variable from numeric

data['Age'] = data.YrSold - data.YearBuilt

data['YearRemodAdd'] = data.YearRemodAdd.eq(data.YearBuilt)

data.drop(['YearBuilt'], inplace=True, axis=1)

data.totBath = data.BsmtFullBath + data.FullBath + 0.5*(data.BsmtHalfBath + data.HalfBath)

data.drop(['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], inplace=True, axis=1)

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], inplace=True, axis=1)

data.TwoKitchens = data.KitchenAbvGr.apply(lambda x: 1 if x>1 else 0)

data.drop(['KitchenAbvGr', 'BedroomAbvGr', 'GarageCond'], inplace=True, axis=1)
numeric_data = data.select_dtypes(include=[np.number])

numeric_data.info()

categorical_data = data.select_dtypes(exclude=[np.number])
categorical_data.info()
data[['MasVnrArea', 'MasVnrType']][data['MasVnrArea'].isna()] #consider no MasVnr (i.e fillna with 0) where MasVnrArea = NA 
ax1 = data.plot(kind='scatter', x='BsmtFinSF1', y='SalePrice', color='r')

ax2 = data.plot(kind='scatter', x='BsmtFinSF2', y='SalePrice', color='y', ax =ax1) 
data['Electrical'].hist()
data[['Fireplaces', 'FireplaceQu']][data['FireplaceQu'].isna()]
numeric_data['MasVnrArea'].fillna(0.0, inplace = True)

categorical_data['Electrical'].fillna('SBrkr', inplace = True)

numeric_data.fillna(numeric_data.median(), inplace=True)

categorical_data.fillna('None', inplace=True)
dummy = pd.get_dummies(categorical_data, drop_first=True)
data = pd.concat([numeric_data, dummy], axis = 1)

data.info()
data_val = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

data_val.info()
val_id = data_val.pop('Id')

data_val.drop(['GarageYrBlt', 'GarageArea', 'TotRmsAbvGrd','MoSold'], inplace=True, axis=1)



data_val.MSSubClass = data_val.MSSubClass.apply(lambda x: str(x)) # converting it to a categorical variable from numeric

data_val['Age'] = data_val.YrSold - data_val.YearBuilt

data_val['YearRemodAdd'] = data_val.YearRemodAdd.eq(data_val.YearBuilt)

data_val.drop(['YearBuilt'], inplace=True, axis=1)

data_val.totBath = data_val.BsmtFullBath + data_val.FullBath + 0.5*(data_val.BsmtHalfBath + data_val.HalfBath)

data_val.drop(['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], inplace=True, axis=1)

data_val['TotalSF'] = data_val['TotalBsmtSF'] + data_val['1stFlrSF'] + data_val['2ndFlrSF']

data_val.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], inplace=True, axis=1)

data_val.TwoKitchens = data_val.KitchenAbvGr.apply(lambda x: 1 if x>1 else 0)

data_val.drop(['KitchenAbvGr', 'BedroomAbvGr', 'GarageCond'], inplace=True, axis=1)
numeric_val = data_val.select_dtypes(include=[np.number])

categorical_val = data_val.select_dtypes(exclude=[np.number])

numeric_val['MasVnrArea'].fillna(0.0, inplace = True)

categorical_val['Electrical'].fillna('SBrkr', inplace = True)

numeric_val.fillna(numeric_val.median(), inplace=True)

categorical_val.fillna('None', inplace=True)



dummy_val = pd.get_dummies(categorical_val, drop_first=True)



data_val = pd.concat([numeric_val, dummy_val], axis = 1)

numeric_val.info()
uncommon_col = list(set(data.columns) ^ set(data_val.columns))

uncommon_col.remove('SalePrice')

print(uncommon_col)

for col in uncommon_col:

    if col in data.columns: data.drop([col], inplace = True, axis = 1)

    if col in data_val.columns: data_val.drop([col], inplace = True, axis = 1)

data.info()
from sklearn.model_selection import train_test_split

np.random.seed(0)

data_train, data_test = train_test_split(data, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()

y_scaler = MinMaxScaler()
y_train = data_train.pop('SalePrice')

x_train = data_train

y_train.head()
numeric_columns = x_train.select_dtypes(include = [np.number]).columns

x_train[numeric_columns] = x_scaler.fit_transform(x_train[numeric_columns])

# y_train =  y_scaler.fit_transform(pd.DataFrame(y_train.iloc[:]))
x_train.describe()
from sklearn.ensemble import RandomForestRegressor

lm = RandomForestRegressor(n_estimators = 1200, max_depth=60,random_state = 0, oob_score = True, n_jobs=10)
lm.fit(x_train, y_train)
y_train_price = lm.predict(x_train)

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
y_test = data_test.pop('SalePrice')

x_test = data_test

x_test[numeric_columns] = x_scaler.transform(x_test[numeric_columns])

# y_test = y_scaler.transform(y_test)
y_pred = lm.predict(x_test)
from sklearn.metrics import mean_squared_error

from math import sqrt

rmsle = sqrt(mean_squared_error(y_test, y_pred))

print('Model RMSLE:',rmsle)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label

# plt.ylim(-0.5,1)

# plt.xlim(-0.5,1)
data_val[numeric_columns] = x_scaler.transform(data_val[numeric_columns])

pred = lm.predict(data_val)
pred = np.expm1(pred)
submission_df = pd.DataFrame({

    'Id': val_id,

    'SalePrice': pred

})
submission_df.head()
submission_df.to_csv('submission_rf_03.csv',index = False)
#Submission score = 0.149