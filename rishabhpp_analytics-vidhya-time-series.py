import pandas as pd

import numpy as np



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error



from catboost import CatBoostRegressor,cv,Pool

import lightgbm as lgb

import xgboost as xgb



import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/janta-hack/train_file (1).csv')

test = pd.read_csv('../input/janta-hack/test_file (1).csv')
train.head()
test.head()
#Count the number of unique values in the dataset

train.nunique()
#Imputing the values of electricity_consumption in test data to 0

test['electricity_consumption'] = 0
train['datetime'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d %H:%M')

test['datetime'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d %H:%M')
df = pd.merge(train,test,on=['ID','datetime','temperature','var1','pressure','windspeed','var2','electricity_consumption'],how='outer')

df.sort_values(by='datetime',inplace=True)

df.head()
df.fillna(0,inplace=True)

scaler = StandardScaler()

df['pressure'] = scaler.fit_transform(df['pressure'].values.reshape(-1,1))

df['windspeed'] = scaler.fit_transform(df['windspeed'].values.reshape(-1,1))
#Extracting date and time features from the dataset

df['hour'] = df.datetime.dt.hour

df['day'] = df.datetime.dt.day

df['month'] = df.datetime.dt.month

df['year'] = df.datetime.dt.year
df.drop('ID',axis=1,inplace=True)

df.set_index('datetime',inplace=True)
plt.rcParams['figure.figsize'] = 12,6
#Visualising the values of temperature,var1,pressure and windspeed wrt time

fig,axs = plt.subplots(4,1,sharex=True)

df['temperature'].plot(figsize=(10,5),title='temp',ax=axs[0])

df['var1'].plot(figsize=(10,5),title='var1',ax=axs[1])

df['pressure'].plot(figsize=(10,5),title='press',ax=axs[2])

df['windspeed'].plot(figsize=(10,5),title='wind',ax=axs[3])

plt.show()
for i in ['temperature','pressure','var1']:

  df[f'cosine_{i}'] = df[i].apply(lambda x: np.cos(x))

  df[f'sine_{i}'] = df[i].apply(lambda x: np.sin(x))
def feature_gen(data):

  for i in ['day','month','year']:

    group = data[['temperature','var1','pressure','windspeed','electricity_consumption',i]].groupby(i).agg(['mean','min','max','std'])

    group.columns = ['_'.join(x)+f'_{i}' for x in group.columns.ravel()]

    data = pd.merge(data,group,on=i,how='left')

  return data
df = feature_gen(df)



df['log_wind'] = np.log1p(df['windspeed'])



group = df[['temperature','var1','pressure','windspeed','var2','day','month','year']].groupby('var2').agg({

    'temperature':['mean','min','max','std'],

    'var1':['mean','min','max','std'],

    'pressure':['mean','min','max','std'],

    'windspeed':['mean','min','max','std'],

    'day':['count'],

    'month':['count'],

    'year':['count']})

group.columns = ['_'.join(x)+'_var2' for x in group.columns.ravel()]

df = pd.merge(df,group,on='var2',how='left')



df['temp_diff_var1'] = abs(df['temperature']) - abs(df['var1'])

df.head()
le = LabelEncoder()

df['var2'] = le.fit_transform(df['var2'])

df['year'] = le.fit_transform(df['year'])
X = df[df['electricity_consumption'] > 0]

X_valid = df[df['electricity_consumption'] == 0]
#XGBRegressor Model

#After cross validation we got the optimum number of trees as 3534

xgbr2 = xgb.XGBRegressor(

 learning_rate =0.1,

 n_estimators=3534,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 nthread=4,

 scale_pos_weight=1,

 seed=27,

 objective='reg:squarederror')



model_xgbr2 = xgbr2.fit(X.drop(['electricity_consumption'],axis=1),X['electricity_consumption'])

y_pred_xgbr2 = xgbr2.predict(X_valid.drop(['electricity_consumption'],axis=1))



test_res = test[['ID']]

test_res = pd.concat([test_res,pd.DataFrame(y_pred_xgbr2,columns=['electricity_consumption'])],axis=1)

test_res.set_index('ID',inplace=True)

test_res.to_csv('sub_xgbr.csv')
#CatBoost Model cross validation

ctr = CatBoostRegressor(iterations=4000,learning_rate=0.1,depth=5,loss_function='RMSE',od_wait=100,od_type='Iter')



#categorical features declaration

cat_feat = np.where(X.drop(['electricity_consumption'],axis=1).dtypes != np.float)[0]



model_ctr = ctr.fit(X.drop(['electricity_consumption'],axis=1),X['electricity_consumption'])

y_pred_ctr = ctr.predict(X_valid.drop(['electricity_consumption'],axis=1))



test_res = test[['ID']]

test_res = pd.concat([test_res,pd.DataFrame(y_pred_ctr,columns=['electricity_consumption'])],axis=1)

test_res.set_index('ID',inplace=True)

test_res.to_csv('sub_ctr.csv')
#Feature Importance plot from CatBoost model

importance = ctr.feature_importances_

fi = pd.Series(index = X.drop(['electricity_consumption'],axis=1).columns, data = importance)

fi.sort_values(ascending=False)[0::][::-1].plot(kind = 'barh',figsize=(10,15))
#LightGBM Model

params = {'num_leaves':31,'max_depth':5,'learning_rate':0.1,'n_estimators':4000,'metric':'mse'}



lgbr1 = lgb.LGBMRegressor(num_leaves=31,max_depth=5,learning_rate=0.1,n_estimators=4000,random_state=27,metric='mse')



#Categorical feature declaration

cat_feat = np.where(X.drop(['electricity_consumption'],axis=1).dtypes != np.float)[0]



#Training data

dtrain = lgb.Dataset(X.drop(['electricity_consumption'],axis=1),label=X['electricity_consumption'])



model_lgbr1 = lgbr1.fit(X.drop(['electricity_consumption'],axis=1),X['electricity_consumption'])

y_pred_lgbr1 = lgbr1.predict(X_valid.drop(['electricity_consumption'],axis=1))



test_res = test[['ID']]

test_res = pd.concat([test_res,pd.DataFrame(y_pred_lgbr1,columns=['electricity_consumption'])],axis=1)

test_res.set_index('ID',inplace=True)

test_res.to_csv('sub_lgbr.csv')
lgb.plot_importance(booster=lgbr1,figsize=(10,10))