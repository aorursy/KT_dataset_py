import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

import numpy as np

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.shape
test.shape
train.head()
test.head()
train.columns
train.SalePrice.describe()
sns.distplot(train.SalePrice)
numeric_columns = ['LotArea' , 'YearBuilt', 'GrLivArea', 'MiscVal', 'GarageArea']
y = train.SalePrice

for col in numeric_columns:

    x=train[col]

    sns.scatterplot(x,y)

    plt.title(col)

    plt.show()
train.corr().SalePrice.sort_values(ascending=False)
categoric_columns = ['Neighborhood','BldgType','OverallQual','TotRmsAbvGrd']





y = train.SalePrice

for col in categoric_columns:

    x=train[col]

    sns.boxplot(x,y)

    plt.title(col)

    plt.show()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

fig, hm = plt.subplots(figsize=(10,8))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

y = train.SalePrice.values



full_df = pd.concat((train,test)).reset_index(drop=True)

full_df.drop(['SalePrice'], axis=1, inplace=True)





missing_values = full_df.isnull().sum().sort_values(ascending=False)



missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)

missing_pct
print(missing_pct.loc[missing_pct.values > 0.8].index)
missing_pct.index
drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']



full_df.drop(drop_cols, axis=1, inplace=True)
none_cols = ['FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType','BsmtCond', 'BsmtExposure', 

             'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'Exterior1st',

             'Exterior2nd']



for col in none_cols:

    full_df[col] = full_df[col].fillna('None')
missing_values = full_df.isnull().sum().sort_values(ascending=False)



missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)

missing_pct.index
zero_cols = ['MasVnrArea', 'BsmtHalfBath','BsmtFullBath','GarageCars','TotalBsmtSF', 

             'GarageArea', 'BsmtUnfSF','BsmtFinSF2', 'BsmtFinSF1','LotFrontage']



for col in zero_cols:

    full_df[col] = full_df[col].fillna(0)

    
mode_cols = ['MSZoning', 'Functional', 'Utilities', 'SaleType',

             'KitchenQual', 'Electrical']



for col in mode_cols:

    mode = full_df[col].mode()

    full_df[col] = full_df[col].fillna(mode[0])
id_na = list(full_df.loc[full_df['GarageYrBlt'].isna()].Id.values)



for row in id_na:

    full_df.loc[row-1,'GarageYrBlt'] = full_df.loc[row-1,'YearBuilt']
missing_values = full_df.isnull().sum().sort_values(ascending=False)



missing_pct = missing_values.loc[missing_values.values > 0]/len(full_df)

missing_pct
full_df['TotalBath'] = full_df['BsmtFullBath'] + full_df['BsmtHalfBath'] + full_df['FullBath'] + full_df['HalfBath']

full_df['TotalArea'] = full_df['TotalBsmtSF'] + full_df['1stFlrSF'] + full_df['2ndFlrSF']

full_df['YrBltAndRemod']=full_df['YearBuilt']+full_df['YearRemodAdd']
full_df['MSSubClass'] = full_df['MSSubClass'].apply(str)



full_df['MoSold'] = full_df['MoSold'].astype(str)
full_df.columns
drop_cols = ['LotFrontage','BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 

             'PoolArea', 'GarageCars','LandSlope', 'MoSold', 'TotalBsmtSF', 'MiscVal',

             'HouseStyle', 'RoofMatl','Condition2'

            ]



full_df.drop(drop_cols, axis=1, inplace=True)
full_df = pd.get_dummies(full_df,drop_first=True)
df_train = full_df[:ntrain]

df_test = full_df[ntrain:]



test_id = test['Id']

df_train.set_index('Id',inplace=True)

df_test.set_index('Id',inplace=True)
df_train = pd.get_dummies(df_train,drop_first=True)

df_test = pd.get_dummies(df_test,drop_first=True)
print(df_train.shape)

print(df_test.shape)
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error

from math import log

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
random_state=42



x_train, x_val, y_train,y_val = train_test_split(df_train, y, random_state=42)
estimator = XGBRegressor(objective='reg:squarederror')



params = {

        'max_depth':range(3,7,2),

        'min_child_weight':range(1,5,2)

         }



    

def tuning(estimator, params):

    grid = GridSearchCV(estimator, param_grid = params, scoring='neg_mean_squared_log_error')

    grid.fit(x_train,y_train)

    print(grid.best_params_)

    print(-grid.best_score_)

tuning(estimator,params)
estimator = XGBRegressor(objective='reg:squarederror',max_depth = 3, min_child_weight = 1)



params = {

        'gamma':[i/10.0 for i in range(0,5)]

         }



tuning(estimator, params)
estimator = XGBRegressor(objective='reg:squarederror',max_depth = 3, min_child_weight = 1,

                         gamma = 0)



params = {

        'learning_rate' : [0.01,0.03,0.1,0.3]

         }



tuning(estimator, params)
estimator = XGBRegressor(objective='reg:squarederror',max_depth =3, min_child_weight = 1)



estimator.fit(x_train,y_train, 

             eval_set=[(x_val, y_val)], verbose=False)



y_pred = estimator.predict(x_val)



print(mean_squared_log_error(y_pred,y_val))



feat_imp = pd.Series(estimator.feature_importances_)
feat = pd.concat([feat_imp,pd.DataFrame(df_train.columns)],axis=1)

feat.columns = ['Importance','Columns']

feat = feat.sort_values(by = 'Importance',ascending=False)

feat.head(25)
feat.tail(60)