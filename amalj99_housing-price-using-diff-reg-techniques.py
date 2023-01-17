# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from vecstack import stacking



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
def missing_vals(df):

    missing=df.isnull().sum()

    missing=missing[missing>0]

    missing.sort_values(inplace=True, ascending=False)

    

    missing.plot.bar(figsize=(12,10))

    plt.xlabel('Columns with missing values')

    plt.ylabel('Number of missing values')

 



missing_vals(train)
def fill_missing_vals(df):

    missing=df.isnull().sum()

    missing=missing[missing > 0]

    for column_name in list(missing.index):  # .df.index puts all the column nmaes in missing variable into single quotes

        if df[column_name].dtype=='object':   # hence easier to call them using []

            df[column_name].fillna(df[column_name].value_counts().index[0], inplace=True)

        elif df[column_name].dtype == 'int64' or 'int16' or 'float64' or 'float16':

            df[column_name].fillna(df[column_name].median(), inplace = True)
fill_missing_vals(train)

train.isnull().sum().max()
missing_vals(test)
fill_missing_vals(test)

test.isnull().sum().max()
train1=train.copy()

test1=test.copy()
def encode(df):

    object_col_ind=[]

    for i in range(df.shape[1]):

        if df.iloc[:,i].dtype =='object':

            object_col_ind.append(i)

        else:

            pass

    label = LabelEncoder()

    for i in object_col_ind:

        df.iloc[:,i]=label.fit_transform(df.iloc[:,i])



encode(train)

encode(test)

print("Train Dtype counts: \n{}".format(train.dtypes.value_counts()))

print("Test Dtype counts: \n{}".format(test.dtypes.value_counts()))
corr_mat = train[["SalePrice","MSSubClass","MSZoning","LotFrontage","LotArea", "BldgType",

                       "OverallQual", "OverallCond","YearBuilt", "BedroomAbvGr", "PoolArea", "GarageArea",

                       "SaleType", "MoSold"]].corr()



f, ax = plt.subplots(figsize=(16, 8))

sns.heatmap(corr_mat, vmax=1 , square=True)
f,ax=plt.subplots(figsize=(14,8))

sns.lineplot(train['YearBuilt'], train['SalePrice'],c='green')
f,ax=plt.subplots(figsize=(14,8))

sns.lineplot(train['OverallQual'], train['SalePrice'],c='red')
f,ax=plt.subplots(figsize=(12,8))

sns.distplot(train['SalePrice'])
X=train.drop('SalePrice', axis=1)

y =train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
def rmse(y,y_pred):

    return np.sqrt(mean_squared_error(np.log(y),np.log(y_pred)))
g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,

                                     max_depth=5, max_features='sqrt',

                                     min_samples_leaf=15, min_samples_split=10,

                                     loss='ls', random_state =2

                                   )



y_pred = cross_val_score(g_boost, X, y, cv=10, n_jobs=-1)

y_pred.mean()
g_boost.fit(X,y)

gbm_pred = g_boost.predict(test)

print(r2_score(g_boost.predict(X),y))

print(rmse(g_boost.predict(X),y))
xg_boost = XGBRegressor( learning_rate=0.05,

                         n_estimators=1000,

                         max_depth=4, min_child_weight=1,

                         gamma=1, subsample=0.9,

                         colsample_bytree=0.2,

                         objective='reg:squarederror', nthread=-1,

                         scale_pos_weight=1, seed=7,

                         reg_alpha=0.00006

                       )





y_pred = cross_val_score(xg_boost, X, y, cv=5, n_jobs=-1)

y_pred.mean()
eval_set=[(X_test,y_test)]  #to prevent overfitting

xg_boost.fit(X,y,eval_set=eval_set,eval_metric='error',verbose=False)

xgb_pred=xg_boost.predict(test)

print(rmse(xg_boost.predict(X_test),y_test))
'''param={'learning_rate':[0.01,0.02,0.04],'max_depth':[3,4,5],'subsample':[0.7,0.8,0.9],

      'gamma':[1,3,5],'n_estimators':[5000,1000,3000]}



gd_cv=GridSearchCV(estimator=xg_boost,param_grid=param,n_jobs=-1,cv=5,scoring='neg_mean_squared_error')

gd_cv.fit(X,y)'''
'''best_parameters = gd_cv.best_params_

print(best_parameters)'''
'''submissionxgb = pd.DataFrame()



submissionxgb['Id'] = np.array(test.index)

submissionxgb['SalePrice'] = xgb_pred

submissionxgb.to_csv('submissionxgb.csv', index=False)'''
random_forest = RandomForestRegressor(n_estimators=1200,

                                      max_depth=15,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                      max_features=None,

                                      random_state=482,

                                      oob_score=True

                                     )



y_pred = cross_val_score(random_forest, X, y, cv=5, n_jobs=-1)

y_pred.mean()
random_forest.fit(X,y)

rf_pred=random_forest.predict(test)

print(rmse(random_forest.predict(X),y))
models=[g_boost,random_forest,xg_boost]
S_train, S_test = stacking(models,

                           X_train, y_train, X_test,

                           regression=True,

                           mode='oof_pred_bag',

                           metric=rmse,

                           n_folds=5,

                           random_state=25,

                           verbose=2

                          )
xgb_lev2 = XGBRegressor(learning_rate=0.05, 

                        n_estimators=500,

                        max_depth=3,

                        n_jobs=-1,

                        random_state=17

                       )



# Fit the 2nd level model on the output of level 1

xgb_lev2.fit(S_train, y_train)
stacked_pred = xgb_lev2.predict(S_test)

print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))
y1_pred_L1 = models[0].predict(test)

y2_pred_L1 = models[1].predict(test)

y3_pred_L1 = models[2].predict(test)

S_test_L1 = np.c_[y1_pred_L1, y2_pred_L1, y3_pred_L1]
test_stacked_pred = xgb_lev2.predict(S_test_L1)
submission = pd.DataFrame()



submission['Id'] = np.array(test.index)

submission['SalePrice'] = test_stacked_pred
submission.to_csv('submission.csv', index=False)
submission