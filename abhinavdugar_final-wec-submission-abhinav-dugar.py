# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/wecrec2020/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import lightgbm as lgb
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,ElasticNetCV,RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import ensemble
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import os
print((os.listdir('/kaggle/input/wecrec2020/')))
df_train = pd.read_csv('/kaggle/input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('/kaggle/input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['Unnamed: 0','F1','F2'], axis = 1, inplace = True)

df_test.drop(['Unnamed: 0','F1','F2'], axis = 1, inplace = True)
df_train.head()
df_test.head()
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']

missing=train_X.isnull().sum()
print(missing)
columns = train_X.columns
#list(columns)
for column in columns:
    print(column)
    print(train_X[column].value_counts())
categorical_columns = ['F3','F4','F5','F7','F8','F9','F11','F12']
numerical_columns = ['F10','F13','F14','F15','F16','F17']
#Tried to Scale the Data but it seemed to increase the RMSE Error 

'''train_X['F6'] = StandardScaler().fit_transform(train_X['F6'].values.reshape(-1, 1))
df_test['F6'] = StandardScaler().fit_transform(df_test['F6'].values.reshape(-1, 1))
train_X['F14'] = StandardScaler().fit_transform(train_X['F14'].values.reshape(-1, 1))
df_test['F14'] = StandardScaler().fit_transform(df_test['F14'].values.reshape(-1, 1))
min_max_scaler = preprocessing.MinMaxScaler()
train_X['F6'] = min_max_scaler.fit_transform(train_X['F6'].values.reshape(-1, 1))
df_test['F6'] = min_max_scaler.fit_transform(df_test['F6'].values.reshape(-1, 1))
train_X['F14'] = min_max_scaler.fit_transform(train_X['F14'].values.reshape(-1, 1))
df_test['F14'] = min_max_scaler.fit_transform(df_test['F14'].values.reshape(-1, 1))'''
#Tried to one-hot encode the data but it increased the RMSE Error 

'''one=OneHotEncoder(handle_unknown='ignore',sparse=False)
one=OneHotEncoder(handle_unknown='ignore',sparse=False)

categorical_train = pd.DataFrame(one.fit_transform(train_X[categorical_columns]))
categorical_test=pd.DataFrame(one.transform(df_test[categorical_columns]))
categorical_train.index=train_X.index
categorical_test.index=df_test.index
print(categorical_train.shape)

numerical_train=train_X[numerical_columns]
numerical_test=df_test[numerical_columns]

X=pd.concat([numerical_train,categorical_train],axis=1)
X_test=pd.concat([numerical_test,categorical_test],axis=1)
print(X.shape)'''
X_train,X_val,y_train,y_val = train_test_split(train_X,train_y,test_size=0.20,random_state=0)
'''xgboost = xgb.XGBRegressor(seed=27)
kfold=StratifiedKFold(n_splits=3,shuffle=True,random_state=12)
gsc=GridSearchCV(xgboost,param_grid={'max_depth':[3,4,5,6,8],'n_estimators':[50,100,200,300,350,500],'learning_rate':[0.1,0.5,0.01,0.05]},scoring='mean_squared_error',cv=kfold,verbose=1,n_jobs=-1)
grid_result=gsc.fit(train_X,train_y)'''
#The best parameters from the above GridSearch:
'''{'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 500}'''
params = {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 500}
final_boost = xgb.XGBRegressor(**params)
final_boost.fit(train_X,train_y)
'''hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 10,
    "num_leaves": 512,  
    "max_bin": 512,
    "num_iterations": 100000,
    "n_estimators": 1000
}
gbm = LGBMRegressor(**hyper_params)
gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        early_stopping_rounds=3000)'''
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 10,
    "num_leaves": 512,  
    "max_bin": 512,
    "num_iterations": 8000,
    "n_estimators": 1200
}
gbm1 = LGBMRegressor(**hyper_params)
gbm1.fit(train_X, train_y)
df_test = df_test.loc[:, 'F3':'F17']
pred_lgbm = gbm1.predict(df_test)
pred_xg = final_boost.predict(df_test)

#Stacking the models
final_pred = (0.8) * pred_lgbm + (0.2) * pred_xg 
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(final_pred)
result.head()
result.to_csv('output.csv', index=False)

#Tried to use a nueral network model but it didnt give out optimum results

'''def model_builder():
    model= tf.keras.Sequential()
    
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
   
    model.add(tf.keras.layers.Dense(units=64, activation='relu',input_shape=[len(X_train.keys())]))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
    
    # une the learning rate for the optimizer 
    #hp_learning_rate=hp.Choice('learning_rate', values=[0.01,0.005])
    
    model.compile(loss=tf.keras.losses.mean_squared_error,optimizer = 'adam'
,metrics= ['mae','mse'])

    return model

model= model_builder()

history= model.fit(
    X_train,y_train,validation_data=(X_val,y_val), epochs=150, batch_size=32,verbose = 1)'''


