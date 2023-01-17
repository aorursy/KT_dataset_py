import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
df_train = pd.read_csv('../input/train.csv')
train_X = df_train.iloc[:,:-1]
train_Y = df_train[['SalePrice']]
train_X
train_Y
df_test =  pd.read_csv('../input/test.csv')
missing_cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
df_train[missing_cols_fillna].dtypes
for col in missing_cols_fillna:
    df_train[col].fillna('None', inplace=True)
    df_test[col].fillna('None', inplace=True)

df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_train.mean(), inplace=True)

df_train.isnull().sum().sum()
df_test.isnull().sum().sum()

import seaborn as sns
#sns.distplot(df_train['SalePrice'])
print("Skewness: %f" % train_Y.skew())
print("kurtosis: %f" % train_Y.kurt())
#df_train['SalePrice'] = np.log(train_Y)
sns.distplot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew())
print("kurtosis: %f" % df_train['SalePrice'].kurt())

categorical_features = df_data.dtypes[df_data.dtypes =="object"].index
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
labelEncoder = LabelEncoder()
df_train[categorical_features]= labelEncoder.fit_transform(categorical_features)
df_test[categorical_features]= labelEncoder.fit_transform(categorical_features)
df_train
train_X = df_train.iloc[:,:-1]
train_Y = df_train[['SalePrice']]
rfg = RandomForestRegressor()
rfg.fit(train_X,train_Y)
predicted_price = rfg.predict(df_test)
print(predicted_price)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_price})
my_submission
my_submission.to_csv('submission.csv', index=False)