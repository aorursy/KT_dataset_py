import os

import sys

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_curve,auc,f1_score,log_loss,accuracy_score

import sklearn

from sklearn.decomposition import PCA 

import json

import requests

import lightgbm as lgb

import optuna

from functools import partial

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error
#LGBM固定パラメータ設定

def init_lgb_params():

    params = {

            'n_jobs':1,

            'task': 'train',

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',      

            'learning_rate': 0.01,

            'n_estimators': 800,

            'random_state':0

     }

    return params
#Optunaモデルlogloss評価

def opt_objective_logloss_groupkfold(X,y,groups,params,trial):



    params_tuning = {

            'num_leaves':trial.suggest_int('num_leaves',10,200),

            'max_depth':trial.suggest_int('max_depth',3,15),        

            'reg_alpha': trial.suggest_uniform('reg_alpha',0, 10),

            'reg_lambda': trial.suggest_uniform('reg_lambda',0, 10),             

            'subsample': trial.suggest_uniform('subsample',0.5, 1.0),  

            'colsample_bytree': trial.suggest_uniform('colsample_bytree',0.1, 1.0)  

    }



    params.update(params_tuning)



    gkf = GroupKFold(n_splits = 5)

    splitter = gkf.split(X,y,groups)



    for train_index, test_index in splitter:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    model=lgb.LGBMRegressor(**params)

    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],eval_metric="rmse",early_stopping_rounds=10,verbose=0)



    y_pred=model.predict(X_test)  

    y_pred = [0 if s<0 else s for s in y_pred]

    

    labels=list(sorted(set(y_test)))

        

    RMSLE=np.sqrt(mean_squared_log_error(y_test, y_pred))

    return RMSLE

        

#    RMSE=np.sqrt(mean_squared_error(y_test, y_pred))    

#    return RMSE
#LGBMパラメータチューニング

def HyperParamTuning(train, key, target, group, ntrial, ncpu):

#    print(train.columns)

    X=train.drop([key,target,group],axis=1)

    y=train[target]

    groups=train[group]



    params=init_lgb_params()

    f=partial(opt_objective_logloss_groupkfold, X, y, groups, params)

    study=optuna.create_study()

    study.optimize(f,n_trials=ntrial,n_jobs=ncpu)

    params.update(study.best_params)

    

    model=lgb.LGBMRegressor(**params)

    

    return model,params
def Prediction(model,train,test,group,key,target):

    X_train=train.drop([key,target,group],axis=1)

    y_train=train[target]    

    X_test=test.drop([key,target,group],axis=1)

    

    model=model.fit(X_train,y_train,eval_metric='rmse')



    y_train_pred=model.predict(X_train)    

    y_test=model.predict(X_test)

    

    y_train_pred = [0 if s<0 else s for s in y_train_pred]

    y_test = [0 if s<0 else s for s in y_test]

#    print(y_train_pred)

    

    RMSLE=np.sqrt(mean_squared_log_error(y_train, y_train_pred))  

    print('RMSLE(train)=',RMSLE)



    return y_test
path='/kaggle/input/exam-for-students20200527'

train_raw=pd.read_csv(os.path.join(path,'train.csv'))

test_raw=pd.read_csv(os.path.join(path,'test.csv'))

city_info=pd.read_csv(os.path.join(path,'city_info.csv'))

station_info=pd.read_csv(os.path.join(path,'station_info.csv'))

data_dict=pd.read_csv(os.path.join(path,'data_dictionary.csv'))

sample_submission=pd.read_csv(os.path.join(path,'sample_submission.csv'))



station_info=station_info.rename(columns={'Station':'NearestStation'})
city_info.head(5)
#station_info.head(5)
train_raw['test_flag']=0

test_raw['test_flag']=1

all=pd.concat([train_raw,test_raw])
all.head(5)
all=pd.merge(all,city_info,how='left',on=['Prefecture','Municipality']).rename(columns={'Latitude':'Latitude_city','Longitude':'Longitude_city'})

all=pd.merge(all,station_info,how='left',on='NearestStation')
all.head(5)
#all.groupby('TimeToNearestStation').count()
#all.groupby(['Prefecture','TimeToNearestStation']).count()
all.tail(5)
all.columns
num_columns=['MinTimeToNearestStation','MaxTimeToNearestStation','Area','AreaIsGreaterFlag',\

            'Frontage','FrontageIsGreaterFlag','TotalFloorArea','TotalFloorAreaIsGreaterFlag',\

            'BuildingYear','PrewarBuilding','Breadth','CoverageRatio','FloorAreaRatio',\

             'Year','Quarter','TradePrice','Latitude','Longitude','Latitude_city','Longitude_city']

ctg_columns=['Type','Region','NearestStation','TimeToNearestStation','FloorPlan','FrontageIsGreaterFlag',\

            'LandShape','Structure','Use','Purpose','Direction','Classification',\

            'CityPlanning','Renovation','Remarks']
all[num_columns]=all[num_columns].fillna(all[num_columns].mean())

all[ctg_columns]=all[ctg_columns].fillna('Z')
all.head(5)
#ラベルエンコーディング

from sklearn import preprocessing

for column in ctg_columns:

    target_column = all[column]

    le = preprocessing.LabelEncoder()

    le.fit(target_column)

    label_encoded_column = le.transform(target_column)

    all[column] = pd.Series(label_encoded_column).astype('category')
all.head(5)
serial_id='id'

target='TradePrice'

#group='Prefecture'

group='NearestStation'
#train=all.query('test_flag==0').drop(['Municipality','NearestStation','DistrictName','test_flag','Remarks'],axis=1)

#test=all.query('test_flag==1').drop(['Municipality','NearestStation','DistrictName','test_flag','Remarks'],axis=1)

train=all.query('test_flag==0').drop(['Prefecture','Municipality','DistrictName','test_flag','Remarks','TimeToNearestStation'],axis=1)

test=all.query('test_flag==1').drop(['Prefecture','Municipality','DistrictName','test_flag','Remarks','TimeToNearestStation'],axis=1)

#train=all.query('test_flag==0').drop(['Prefecture','Municipality','DistrictName','test_flag','TimeToNearestStation'],axis=1)

#test=all.query('test_flag==1').drop(['Prefecture','Municipality','DistrictName','test_flag','TimeToNearestStation'],axis=1)
train.head(5)
train.columns
model,params=HyperParamTuning(train, serial_id, target, group, 20, 5)
y_pred=Prediction(model, train, test, group, serial_id, target)
#y_pred
#y_pred = [0 if s<0 else s for s in y_pred ]
#sample_submission
submission=sample_submission.copy(deep=True)

submission['TradePrice']=y_pred
train['TradePrice'].describe()
submission['TradePrice'].describe()
submission.columns
submission.to_csv('submision_late_1st.csv',index=False)