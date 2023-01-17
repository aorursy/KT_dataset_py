import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(5)
test_df.head(5)
train_df.shape
test_df.shape
data = pd.concat([train_df,test_df])
data.shape


data.head()
plt.figure(figsize=(15,8))

sns.heatmap(data.isnull(), yticklabels = False, cmap = 'summer')
data.isnull().sum().sort_values(ascending=False)[0:32]

data.drop(['PoolQC','MiscFeature','Alley','Fence','Id'], axis=1, inplace=True)

data.info()
data.isnull().sum().sort_values(ascending=False)[0:32]

print("No. of columns containing null values")

print(len(data.columns[data.isna().any()]))
obj_col = ['MSZoning','Utilities','LotFrontage','Exterior2nd','Exterior1st','MasVnrType','BsmtCond', 'BsmtQual','BsmtExposure','BsmtFinType2','BsmtFinType1' ,'Electrical', 'KitchenQual', 'GarageType','FireplaceQu'

          ,'GarageFinish', 'GarageQual','GarageCond' ,'SaleType','Functional'  ]









for item in obj_col:

    data[item] = data[item].fillna(data[item].mode()[0])

num_col = ['SalePrice','GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars']

for item in num_col:

    data[item] = data[item].fillna(data[item].mean())
data = pd.get_dummies(data, drop_first=True)
data.isnull().sum().sort_values(ascending=False)[0:10]

data.shape
data.head()
data_train = data.iloc[:1460, :]

data_test = data.iloc[1460:, :]
from sklearn.model_selection import train_test_split
X_train= data_train.drop('SalePrice', axis=1).values

y_train= data_train['SalePrice'].values



##X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)





X_test= data_test.drop('SalePrice', axis=1).values

from sklearn.ensemble import RandomForestClassifier

rdf= RandomForestClassifier()
rdf.fit(X_train,y_train)
y_test_predict = rdf.predict(X_test)
y_test_predict
#output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_test_predict})

#output.to_csv('house_sales_prediction_simple.csv', index=False)

#print("Done!")
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

regressor = XGBRegressor()

regressor.fit(X_train, y_train)
y_train_predict = regressor.predict(X_train)

print(r2_score(y_train,y_train_predict))
y_test_predict = regressor.predict(X_test)
output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_test_predict})

output.to_csv('house_sales_prediction_simple.csv', index=False)

print("Done!")