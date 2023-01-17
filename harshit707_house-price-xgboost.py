import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

#Read the data

df_train = pd.read_csv("../input/iowa-house-prices/train.csv", index_col = 'Id')

df_test = pd.read_csv("../input/iowa-house-prices/test.csv", index_col ='Id')
df_train.head()
df_train.isna().sum()
df_train.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = df_train['SalePrice']
y 
df_train.drop(axis = 1, labels = ['SalePrice'], inplace = True)
# Shape

print(df_train.shape)

print(df_test.shape)
df_train.columns
#dropping columns which are not required for prediction of new house

col_drop = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

df_train.drop(labels = col_drop, axis = 1, inplace = True)

df_test.drop(labels = col_drop,axis=1,inplace=True)
print(df_train.shape)

print(df_test.shape)
num_col = [col for col in df_train.columns if df_train[col].dtype in ['int64','float64']]
num_col
cols_nulls = df_train[num_col].isnull().sum()
cols_nulls[cols_nulls >0]
#we can drop LotFrontage is have many null values

df = df_train[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].dropna(axis=0)

plt.figure(figsize=(15,5))

sns.distplot(df['LotFrontage'],bins=30,kde=True)


plt.figure(figsize = (12,4))

sns.distplot(a = df['MasVnrArea'], bins = 30, norm_hist=False, kde=True, color = 'purple')


plt.figure(figsize = (12,4))

sns.distplot(a = df['GarageYrBlt'], bins = 30, norm_hist=False, kde=True, color = 'orange')
from sklearn.impute import SimpleImputer



numerical_cols_median = ['LotFrontage']

numerical_transformer_median = SimpleImputer(strategy = 'median')



numerical_cols_mod = ['MasVnrArea']

numerical_transformer_mod = SimpleImputer(strategy = 'most_frequent')



numerical_cols_mean = ['GarageYrBlt']

numerical_transformer_mean = SimpleImputer(strategy = 'mean')



numerical_cols_remain = set(num_col) - set(numerical_cols_mean) - set(numerical_cols_median) - set(numerical_cols_mod)

numerical_cols_remain = list(numerical_cols_remain)
cat_data = [col for col in df_train.columns if df_train[col].dtype == 'object']
cat_data 
cat_null = df_train[cat_data].isnull().sum()

cat_null[cat_null > 0]
#Because there are a few columns with too many missing values, we'll filter them out from the data

df_train.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
#Because there are a few columns with too many missing values, we'll filter them out from the data

df_test.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
#Redo categorical_cols variable

categorical_cols = [col for col in df_train.columns if df_train[col].dtype == 'object']
#performing imputer and OHE to encode the features

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
cat_tranform = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),

                              ('onehot',OneHotEncoder(handle_unknown='ignore'))])
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num_median', numerical_transformer_median, numerical_cols_median),

                                               ('num_mod', numerical_transformer_mod, numerical_cols_mod),

                                               ('num_mean', numerical_transformer_mean, numerical_cols_mean),

                                               ('num_rest', numerical_transformer_mean, numerical_cols_remain),

                                              ('cat', cat_tranform, categorical_cols)])
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn import model_selection, metrics

from sklearn.model_selection import GridSearchCV
X_train, X_valid, y_train, y_valid = train_test_split(df_train, y, test_size = 0.3, random_state = 0)

def modelfit(model):

    pipeline = Pipeline(steps=[('preprocessing',preprocessor),

                              ('model',model)])

    pipeline.fit(X_train,y_train)

    pred = pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid,pred)

    print('MAE',mae)
xgb1 = XGBRegressor( learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,

                     colsample_bytree=0.8, nthread=4, scale_pos_weight=1, seed=27, objective='reg:squarederror')

modelfit(xgb1)
from sklearn.model_selection import RandomizedSearchCV

# param_1= {'max_depth' : range(3, 10, 2),

#           'min_child_weight' : range(1, 6, 2),

#           'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100],

#           'gamma' : [i/10.0 for i in range(0,5)],

#           'subsample' : [i/10.0 for i in range(6,10)],

#           'colsample_bytree' : [i/10.0 for i in range(6,10)]}

#Use parameters and apply tunning

xgb_final = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=0.0,

             importance_type='gain', learning_rate=0.005, max_delta_step=0,

             max_depth=4, min_child_weight=1, missing=None, n_estimators=10000,

             n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

             reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=27,

             silent=None, subsample=0.8, verbosity=1)

modelfit(xgb_final)

#my model with all tunings

my_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=0.0,

             importance_type='gain', learning_rate=0.005, max_delta_step=0,

             max_depth=4, min_child_weight=1, missing=None, n_estimators=5000,

             n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

             reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=27,

             silent=None, subsample=0.8, verbosity=1)

pipeline_final = Pipeline(steps=[('preprocessor', preprocessor),

                                ('model', my_model)])



pipeline_final.fit(df_train, y)



preds_test = pipeline_final.predict(df_test)

output = pd.DataFrame({'Id': df_test.index, 'SalePrice' : preds_test})
output