import numpy as np 
import pandas as pd 
import pandas_profiling
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
import xgboost as xgboost

%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting

import os
print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')
df_train.head(10)
df_test = pd.read_csv('../input/test.csv')
df_test.head()
#pandas_profiling.ProfileReport(df_train)
df_train.info()
def fill_null(df):
    df["3SsnPorch"] = df["3SsnPorch"].fillna(0)
    df["Alley"] = df["Alley"].fillna("None")
    df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(0)
    df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0)
    df["EnclosedPorch"] = df["EnclosedPorch"].fillna(0)
    df["Fence"] = df["Fence"].fillna("None")
    df["HalfBath"] = df["HalfBath"].fillna(0)
    df["LowQualFinSF"] = df["LowQualFinSF"].fillna(0)
    df["MiscFeature"] = df["MiscFeature"].fillna("None")
    df["MiscVal"] = df["MiscVal"].fillna(0)
    df["PoolArea"] = df["PoolArea"].fillna(0)
    df["ScreenPorch"] = df["ScreenPorch"].fillna(0)
    return df
    
df_train = fill_null(df_train)
df_test = fill_null(df_test)
ids_train = df_train['Id']
df_train = df_train.drop('Id',1)

ids_test = df_test['Id']
df_test = df_test.drop('Id',1)
def get_cat_cols(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
    newdf = df.select_dtypes(exclude=numerics)
    return newdf.columns

def get_num_cols(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
    newdf = df.select_dtypes(include=numerics)
    return newdf.columns
def fill(df):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df[get_num_cols(df)])
    array_transform = imp_mean.transform(df[get_num_cols(df)])
    df[get_num_cols(df)] = array_transform

    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_most_frequent.fit(df[get_cat_cols(df)])
    array_transform = imp_most_frequent.transform(df[get_cat_cols(df)])
    df[get_cat_cols(df)] = array_transform
    
    return df

df_train = fill(df_train)
df_test = fill(df_test)
#выбросы
for col in get_num_cols(df_train):
    fig, ax = plt.subplots()
    ax.scatter(x = df_train[col], y = df_train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(col, fontsize=13)
    plt.show()
#удалим строки имеющие выбросы по каким-то признакам
df_train = df_train.drop(df_train[(df_train['LotFrontage']>300) & (df_train['SalePrice']<300000)].index)
df_train = df_train.drop(df_train[(df_train['LotArea']>150000) & (df_train['SalePrice']<400000)].index)
df_train = df_train.drop(df_train[(df_train['YearBuilt']<2000) & (df_train['SalePrice']>700000)].index)
df_train = df_train.drop(df_train[(df_train['YearRemodAdd']<2000) & (df_train['SalePrice']>700000)].index)
df_train = df_train.drop(df_train[(df_train['MasVnrArea']>1250) & (df_train['SalePrice']<700000)].index)
df_train = df_train.drop(df_train[(df_train['BsmtFinSF2']>1300) & (df_train['SalePrice']<400000)].index)
df_train = df_train.drop(df_train[(df_train['WoodDeckSF']>800) & (df_train['SalePrice']<500000)].index)
## выделим целевую переменную и преобразуем к логарифму
y_train = np.log(df_train['SalePrice'])
df_train = df_train.drop('SalePrice',1)
def do_scalar(df):
    scaler = StandardScaler()
    scaler.fit(df[get_num_cols(df)])
    df[get_num_cols(df)] = scaler.transform(df[get_num_cols(df)])
    return df
    
df_train = do_scalar(df_train)
df_test = do_scalar(df_test)
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop(column_name,1)
    return df
for col in get_cat_cols(df_train):
    df_train = create_dummies(df_train,col)
    df_test = create_dummies(df_test,col)
def del_corr(df):
    df = df.drop('Exterior2nd_CBlock',1)
    df = df.drop('Exterior2nd_CmentBd',1)
    df = df.drop('Exterior2nd_MetalSd',1)
    df = df.drop('Exterior2nd_VinylSd',1)
    df = df.drop('SaleType_New',1)
    return df

df_train = del_corr(df_train)
df_test = del_corr(df_test)
for col in df_train.columns:
    if not col in df_test.columns:
        df_test[col]=0
for col in df_test.columns:
    if not col in df_train.columns:
         df_test = df_test.drop(col,1)
ridgeCV = linear_model.RidgeCV()
ridgeCV.fit(df_train[df_train.columns], y_train)
y_pred = ridgeCV.predict(df_test[df_test.columns])
submission_df = {"Id": ids_test,
                 "SalePrice": y_pred}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_ridge.csv', index=False)
regrRFR = RandomForestRegressor()
regrRFR.fit(df_train[df_train.columns], y_train)
y_pred = regrRFR.predict(df_test[df_test.columns])
submission_df = {"Id": ids_test,
                 "SalePrice": np.e**y_pred}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_RF.csv', index=False)
#gbrt = GradientBoostingRegressor()
#param_grid = {'max_depth': [2,3,4],'n_estimators': [100,150,200,500,700,1000],'learning_rate':[0.01,0.05, 0.1, 0.3], 'loss': ['ls', 'lad', 'huber', 'quantile']}
#model = GridSearchCV(estimator=gbrt, param_grid=param_grid, n_jobs=-1, cv=7, scoring='neg_mean_squared_error')
#model.fit(df_train[df_train.columns], y_train)

#print('GradientBoostingRegressor...')
#print('Best Params:')
#print(model.best_params_)
#print('Best CV Score:')
#print(-model.best_score_)
#GradientBoostingRegressor...
#Best Params:
#{'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 2, 'n_estimators': 1000}
#Best CV Score:
#0.014604507039396397
gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=1000,learning_rate=0.1, loss = 'huber')
gbrt.fit(df_train[df_train.columns], y_train)
y_pred = gbrt.predict(df_test[df_test.columns])
submission_df = {"Id": ids_test,
                 "SalePrice": np.e**y_pred}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_GBR.csv', index=False)
#param_grid = {
#    'max_depth': [2, 3, 4],
#    'n_estimators': [200, 500, 1000, 1500],
#    'learning_rate': [0.005, 0.01, 0.025, 0.1]
#}
#xgb = xgboost.XGBRegressor(nthread=-1)
#model = GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=-1, cv=7, scoring='neg_mean_squared_error')
#model.fit(df_train[df_train.columns].as_matrix(), y_train)

#print('XGBRegressor...')
#print('Best Params:')
#print(model.best_params_)
#print('Best CV Score:')
#print(-model.best_score_)
#/opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:8: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  
#XGBRegressor...
#Best Params:
#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}
#Best CV Score:
#0.014560327485736105
xgb = xgboost.XGBRegressor(nthread=-1, learning_rate=0.1, max_depth=3, n_estimators=500)
xgb.fit(df_train[df_train.columns].as_matrix(), y_train)
y_pred = xgb.predict(df_test[df_test.columns].as_matrix())
submission_df = {"Id": ids_test,
                 "SalePrice": np.e**y_pred}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_xgb.csv', index=False)