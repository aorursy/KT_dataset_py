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
import xgboost as xgb

import lightgbm as lgb

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error as mse

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train['price'] = np.log1p(df_train['price'])
df_train.loc[df_train['sqft_living'] > 13000]
df_train.loc[(df_train['price']>12) & (df_train['grade'] == 3)]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train = df_train.loc[df_train['id']!=8912]

df_train = df_train.loc[df_train['id']!=2302]

df_train = df_train.loc[df_train['id']!=4123]

df_train = df_train.loc[df_train['id']!=7173]

df_train = df_train.loc[df_train['id']!=2775]
skew_columns = ['bedrooms','sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['date'] = df['date'].astype('int')

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)
df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

X_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

X_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')

y_train = X_train['price']

del X_train['id']

del X_train['price']

del X_train['per_price']

del X_test['id']
X_train.columns, X_train.shape, X_test.columns, X_test.shape
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.5,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



print('Transform DMatrix...')

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)
model = xgb.train(xgb_params, dtrain, num_boost_round = rounds)

preds = model.predict(dtest)

preds = np.exp(model.predict(dtest))
x_sample,x_unseen,y_sample,y_unseen=train_test_split(X_train,y_train,test_size=1/5)

    

model_lgb=lgb.LGBMRegressor(

                           learning_rate=0.05,

                           n_estimators=1000,

                           subsample=0.8,

                           colsample_bytree=0.5,

                           reg_alpha=0.2,

                           reg_lambda=10,

                           num_leaves=30,

                           silent=True,                                                      

                           )

model_lgb.fit(x_sample,y_sample,eval_set=(x_unseen,y_unseen),verbose=0,early_stopping_rounds=1000,

             eval_metric='rmse')



lgb_score=mse(np.exp(model_lgb.predict(x_unseen)),np.exp(y_unseen))**0.5

print("RMSE unseen : {}".format(lgb_score))







lgb_pred=np.exp(model_lgb.predict(X_test))





fig, ax = plt.subplots(figsize=(10,10))

lgb.plot_importance(model_lgb, ax=ax)

plt.show()
predict = (preds + lgb_pred)/2
sub=pd.read_csv('../input/sample_submission.csv')

sub['price']=predict
submit = pd.DataFrame({'id': df_test['id'], 'price': predict})

submit.head()
submit.to_csv('xgb_lgb_sub.csv',index=False)