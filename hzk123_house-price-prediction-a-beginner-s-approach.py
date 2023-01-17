

import numpy as np 

import pandas as pd 





import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(df_train['SalePrice'])
df_train['SalePrice_log'] = np.log(df_train['SalePrice'])

df_train.drop('SalePrice', axis=1, inplace=True)
plt.figure(figsize=(15,9))

sns.heatmap(df_train.isnull())
total = df_train.isnull().sum().sort_values(ascending = False)

percentage = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percentage], axis=1, keys=['total', 'percent'])

df_train = df_train.drop((missing_data[missing_data['total']>81]).index,1)

df_train.isnull().sum().sort_values(ascending=False)
total_test = df_test.isnull().sum().sort_values(ascending = False)

percentage_test = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test, percentage_test], axis=1, keys=['total', 'percent'])

df_test = df_test.drop((missing_data_test[missing_data_test['total']>78]).index,1)

df_test.isnull().sum().sort_values(ascending = False)
train_num_cols = df_train.select_dtypes(exclude='object').columns

train_cat_cols = df_train.select_dtypes(include='object').columns
test_num_cols = df_test.select_dtypes(exclude='object').columns

test_cat_cols = df_test.select_dtypes(include='object').columns
for i in range(0, len(train_num_cols)):

    df_train[train_num_cols[i]] = df_train[train_num_cols[i]].fillna(df_train[train_num_cols[i]].mean())



for i in range(0, len(test_num_cols)):

    df_test[test_num_cols[i]] = df_test[test_num_cols[i]].fillna(df_test[test_num_cols[i]].mean()) 
for i in range(0, len(train_cat_cols)):

    df_train[train_cat_cols[i]] = df_train[train_cat_cols[i]].fillna(df_train[train_cat_cols[i]].mode()[0])



for i in range(0, len(test_cat_cols)):

    df_test[test_cat_cols[i]] = df_test[test_cat_cols[i]].fillna(df_test[test_cat_cols[i]].mode()[0])
plt.figure(figsize=(20,9))

sns.heatmap(df_train.corr())
df_train.drop(['MasVnrArea', 'GarageYrBlt','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea',

         'BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

         'MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1, inplace=True)

df_train['GrLivArea_log'] = np.log(df_train['GrLivArea'])

df_train.drop('GrLivArea', axis=1, inplace=True)
df_test['GrLivArea_log'] = np.log(df_test['GrLivArea'])

df_test.drop('GrLivArea', axis=1, inplace=True)
df_test.drop(['MasVnrArea', 'GarageYrBlt','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea',

         'BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

         'MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1, inplace=True)
train_cat_cols1 = df_train.select_dtypes(include='object').columns

nrows = 13

ncols = 3



fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*3))



for r in range(0, nrows):

    for c in range(0, ncols):

        i = r*ncols + c

        if i < len(train_cat_cols1):

            sns.boxplot(x=train_cat_cols1[i], y= 'SalePrice_log', data= df_train, ax= ax[r][c])



plt.tight_layout()

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train[train_cat_cols1] = df_train[train_cat_cols1].apply(lambda col:le.fit_transform(col.astype(str)))
test_cat_cols1 = df_test.select_dtypes(include='object').columns

df_test[test_cat_cols1] = df_test[test_cat_cols1].apply(lambda col:le.fit_transform(col.astype(str)))
X_train = df_train.drop('SalePrice_log', axis=1)

y_train = df_train['SalePrice_log']

X_test = df_test
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)
lm.score(X_train, y_train)
predictions_lm = lm.predict(X_test)

predictions_lm = np.exp(predictions_lm)

final_predict = predictions_lm
sub_file = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub_file.head()
sub_file['SalePrice'] = final_predict
sub_file.head()
sub_file.to_csv('final_submission1.csv', index=False)