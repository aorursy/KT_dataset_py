# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# General imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import gc
from tqdm import tqdm
# SKLearn imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OrdinalEncoder

# TensorFlow imports
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import tensorflow.keras.layers as KL
# Other ML packages 
from scipy.optimize import curve_fit
import xgboost as xgb
from lightgbm import LGBMRegressor


# Input file
Num_lag=28
Hessian_Flag=True
STD_Flag=True
Num_bags=3
def Preprocess(Num_lag,Hessian_Flag,Num_bags=3):
    train = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
    train['Region']=train.Country_Region+train.Province_State.fillna('')
    List_of_Features_base=['ConfirmedCases','Fatalities','ConfirmedCases_Move_Avg','Fatalities_Move_Avg','Diff_ConfirmedCases_Move_Avg',
                      'Diff_Fatalities_Move_Avg']
    target_list=[]
    if Hessian_Flag: 
        List_of_Features_base.append('Hess_ConfirmedCases_Move_Avg')
        List_of_Features_base.append('Hess_ConfirmedCases')
        List_of_Features_base.append('Hess_Fatalities_Move_Avg')
        List_of_Features_base.append('Hess_Fatalities')
    #Log normalization
    #train['ConfirmedCases_temp']=np.log1p(train['ConfirmedCases'])
    #train['Fatalities_temp']=np.log1p(train['Fatalities'])
    #train = train.drop('ConfirmedCases', 1)
    #train = train.drop('Fatalities', 1)
    #train = train.rename(columns={'ConfirmedCases_temp': 'ConfirmedCases', 'Fatalities_temp': 'Fatalities'})
    #First time derivative (gradient)
    train['Diff_ConfirmedCases']=train.groupby('Region')[['ConfirmedCases']].diff()
    train['Diff_Fatalities']=train.groupby('Region')[['Fatalities']].diff()
    #Second time derivative (Hessian)
    if Hessian_Flag:
        train['Hess_ConfirmedCases']=train.groupby('Region')[['Diff_ConfirmedCases']].diff()
        train['Hess_Fatalities']=train.groupby('Region')[['Diff_Fatalities']].diff()    
    #lagged features avarges
    train['ConfirmedCases_Move_Avg']=train['ConfirmedCases']
    train['Fatalities_Move_Avg']=train['Fatalities']
    
    train['Diff_Fatalities_Move_Avg']=train['Diff_Fatalities']
    train['Diff_ConfirmedCases_Move_Avg']=train['Diff_ConfirmedCases']
    if Hessian_Flag:
        train['Hess_Fatalities_Move_Avg']=train['Hess_Fatalities']
        train['Hess_ConfirmedCases_Move_Avg']=train['Hess_ConfirmedCases']
    for bag in range(1,Num_bags+1):  
        List_of_Features_temp=List_of_Features_base
        for i in range(bag,Num_bags*Num_lag+1,Num_bags):
        # Confirmed lags
            train['ConfirmedCases_Lag_'+str(i)] = train.groupby('Region')[['ConfirmedCases']].shift(i)
            List_of_Features_temp.append('ConfirmedCases_Lag_'+str(i))
        # Confirmed derivatives lags
            train['Diff_ConfirmedCases_Lag_'+str(i)] = train.groupby('Region')[['Diff_ConfirmedCases']].shift(i)   
            List_of_Features_temp.append('Diff_ConfirmedCases_Lag_'+str(i))
            if Hessian_Flag:
                train['Hess_ConfirmedCases_Lag_'+str(i)] = train.groupby('Region')[['Hess_ConfirmedCases']].shift(i) 
                List_of_Features_temp.append('Hess_ConfirmedCases_Lag_'+str(i))
                train['Hess_Fatalities_Lag_'+str(i)] = train.groupby('Region')[['Hess_Fatalities']].shift(i)
                List_of_Features_temp.append('Hess_Fatalities_Lag_'+str(i))
        # Fatalities lags
            train['Fatalities_Lag_'+str(i)] = train.groupby('Region')[['Fatalities']].shift(i)
            List_of_Features_temp.append('Fatalities_Lag_'+str(i))
        # Fatalities derivatives lags
            train['Diff_Fatalities_Lag_'+str(i)] = train.groupby('Region')[['Diff_Fatalities']].shift(i)  
            List_of_Features_temp.append('Diff_Fatalities_Lag_'+str(i))

        #avaraged column
            train['ConfirmedCases_Move_Avg'] = train['ConfirmedCases_Move_Avg']+train['ConfirmedCases_Lag_'+str(i)]
            train['Fatalities_Move_Avg'] = train['Fatalities_Move_Avg']+train['Fatalities_Lag_'+str(i)]
            train['Diff_Fatalities_Move_Avg']=train['Diff_Fatalities_Move_Avg']+train['Diff_Fatalities_Lag_'+str(i)]
            train['Diff_ConfirmedCases_Move_Avg']=train['Diff_ConfirmedCases_Move_Avg']+train['Diff_ConfirmedCases_Lag_'+str(i)]
            if Hessian_Flag:
                train['Hess_Fatalities_Move_Avg']=train['Hess_Fatalities_Move_Avg']+train['Hess_Fatalities_Lag_'+str(i)]
                train['Hess_ConfirmedCases_Move_Avg']=train['Hess_ConfirmedCases_Move_Avg']+train['Hess_ConfirmedCases_Lag_'+str(i)]     
        
    # Moving avarage block    
        train['ConfirmedCases_Move_Avg']=train['ConfirmedCases_Move_Avg']/(Num_lag+1)
        train['Fatalities_Move_Avg']=train['Fatalities_Move_Avg']/(Num_lag+1)
        train['Diff_ConfirmedCases_Move_Avg']=train['Diff_ConfirmedCases_Move_Avg']/(Num_lag+1) 
        train['Diff_Fatalities_Move_Avg']=train['Diff_Fatalities_Move_Avg']/(Num_lag+1)
        if Hessian_Flag:
            train['Hess_ConfirmedCases_Move_Avg']=train['Hess_ConfirmedCases_Move_Avg']/(Num_lag+1) 
            train['Hess_Fatalities_Move_Avg']=train['Hess_Fatalities_Move_Avg']/(Num_lag+1) 
        target_list.append(List_of_Features_temp)
    train['serd']=train.groupby('Region').cumcount()
    train.loc[train.ConfirmedCases==0,'days_since_confirmed']=0
    train.loc[train.ConfirmedCases>0,'days_since_confirmed']=train[train.ConfirmedCases>0].groupby('Region').cumcount()
    return(train,target_list)
    
train,List_of_Features=Preprocess(Num_lag,Hessian_Flag,Num_bags=3)
List_of_Features
lgbm_cc=LGBMRegressor(num_leaves = 85,learning_rate =10**-1.89,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.1),min_child_samples =2,
                      subsample =0.97,subsample_freq=10,colsample_bytree = 0.68,reg_lambda=10**1.4,random_state=1234,n_jobs=4)
lgbm_f=LGBMRegressor(num_leaves = 26,learning_rate =10**-1.63,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.04),min_child_samples =14,
                     subsample =0.66,subsample_freq=5,colsample_bytree = 0.8,reg_lambda=10**1.92,random_state=1234,n_jobs=4)


oe=OrdinalEncoder()
X=oe.fit_transform(train[['Country_Region','Province_State']].fillna(''))
train['CR']=X[:,0]
train['PS']=X[:,1]
Feature_list=List_of_Features[2]
Feature_list.append('CR')
Feature_list.append('PS')
Feature_list.append('days_since_confirmed')
lgbm_cc.fit(train.loc[:,Feature_list],train.Diff_ConfirmedCases,categorical_feature=['CR','PS'])
lgbm_f.fit(train.loc[:,Feature_list],train.Diff_Fatalities,categorical_feature=['CR','PS'])   
def Postprocess(Feature_list,Num_lag,Hessian_Flag):
    test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
    test['Region']=test.Country_Region+test.Province_State.fillna('')
    Prediction = pd.concat((train,test[test.Date>train.Date.max()])).reset_index(drop=True)
    Prediction.sort_values(['Country_Region','Province_State','Date'],inplace=True)
    XP=oe.transform(Prediction[['Country_Region','Province_State']].fillna(''))
    Prediction['CR']=XP[:,0]
    Prediction['PS']=XP[:,1]
    Prediction['serd']=Prediction.groupby('Region').cumcount()
    Prediction.loc[Prediction.ConfirmedCases.isnull(),'ConfirmedCases']=1 #Heuristic
    Prediction.loc[Prediction.ConfirmedCases==0,'days_since_confirmed']=0
    Prediction.loc[Prediction.ConfirmedCases>0,'days_since_confirmed']=Prediction[Prediction.ConfirmedCases>0].groupby('Region').cumcount() 
    #Prediction['ConfirmedCases']=np.log1p(Prediction['ConfirmedCases'])
    #Prediction['Fatalities']=np.log1p(Prediction['Fatalities'])
    Prediction['Diff_ConfirmedCases']=Prediction.groupby('Region')[['ConfirmedCases']].diff()
    Prediction['Diff_Fatalities']=Prediction.groupby('Region')[['Fatalities']].diff()
    for serd in range(train.serd.max()+1,Prediction.serd.max()+1):
        print(serd)
        Prediction['Diff_Fatalities_Move_Avg']=0
        Prediction['Diff_ConfirmedCases_Move_Avg']=0
        Prediction['Hess_Fatalities_Move_Avg']=0
        Prediction['Hess_ConfirmedCases_Move_Avg']=0
        for i in range(1,2*Num_lag+1,2): 
            Prediction['ConfirmedCases_Lag_'+str(i)] = Prediction.groupby('Region')[['ConfirmedCases']].shift(i)
            Prediction['Diff_ConfirmedCases_Lag_'+str(i)] = Prediction.groupby('Region')[['Diff_ConfirmedCases']].shift(i)
            Prediction['Fatalities_Lag_'+str(i)] = Prediction.groupby('Region')[['Fatalities']].shift(i)
            Prediction['Diff_Fatalities_Lag_'+str(i)] = Prediction.groupby('Region')[['Diff_Fatalities']].shift(i)  
            Prediction['ConfirmedCases_Move_Avg'] = train['ConfirmedCases_Move_Avg']+train['ConfirmedCases_Lag_'+str(i)]
            Prediction['Fatalities_Move_Avg'] = train['Fatalities_Move_Avg']+train['Fatalities_Lag_'+str(i)]
            Prediction['Diff_Fatalities_Move_Avg']=train['Diff_Fatalities_Move_Avg']+train['Diff_Fatalities_Lag_'+str(i)]
            Prediction['Diff_ConfirmedCases_Move_Avg']=train['Diff_ConfirmedCases_Move_Avg']+train['Diff_ConfirmedCases_Lag_'+str(i)]

            if Hessian_Flag:
                Prediction['Hess_ConfirmedCases_Lag_'+str(i)] = Prediction.groupby('Region')[['Hess_ConfirmedCases']].shift(i) 
                Prediction['Hess_Fatalities_Lag_'+str(i)] = Prediction.groupby('Region')[['Hess_Fatalities']].shift(i)
                Prediction['Hess_Fatalities_Move_Avg']=train['Hess_Fatalities_Move_Avg']+train['Hess_Fatalities_Lag_'+str(i)]
                Prediction['Hess_ConfirmedCases_Move_Avg']=train['Hess_ConfirmedCases_Move_Avg']+train['Hess_ConfirmedCases_Lag_'+str(i)] 
                
        Prediction['ConfirmedCases_Move_Avg']=Prediction['ConfirmedCases_Move_Avg']/Num_lag
        Prediction['Fatalities_Move_Avg']=Prediction['Fatalities_Move_Avg']/Num_lag    
        Prediction['Diff_ConfirmedCases_Move_Avg']=Prediction['Diff_ConfirmedCases_Move_Avg']/Num_lag      
        Prediction['Diff_Fatalities_Move_Avg']=Prediction['Diff_Fatalities_Move_Avg']/Num_lag
        if Hessian_Flag:
            Prediction['Hess_ConfirmedCases_Move_Avg']=Prediction['Hess_ConfirmedCases_Move_Avg']/Num_lag
            Prediction['Hess_Fatalities_Move_Avg']=Prediction['Hess_Fatalities_Move_Avg']/Num_lag
        Prediction.loc[Prediction.serd==serd,'Diff_ConfirmedCases']= lgbm_cc.predict(Prediction.loc[Prediction.serd==serd,Feature_list])
        Prediction.loc[(Prediction.serd==serd) & (Prediction.Diff_ConfirmedCases<0),'Diff_ConfirmedCases']=0
        Prediction.loc[Prediction.serd==serd,'ConfirmedCases']=Prediction.loc[Prediction.serd==serd,'Diff_ConfirmedCases']+Prediction.loc[Prediction.serd==serd,'ConfirmedCases_Lag_1']
        #Prediction.loc[Prediction.serd==serd,'ConfirmedCases']=np.exp(Prediction.loc[Prediction.serd==serd,'ConfirmedCases'])-1
        Prediction.loc[Prediction.serd==serd,'Diff_Fatalities']= lgbm_f.predict(Prediction.loc[Prediction.serd==serd,Feature_list])
        Prediction.loc[(Prediction.serd==serd) & (Prediction.Diff_Fatalities<0),'Diff_Fatalities']=0
        Prediction.loc[Prediction.serd==serd,'Fatalities']=Prediction.loc[Prediction.serd==serd,'Diff_Fatalities']+Prediction.loc[Prediction.serd==serd,'Fatalities_Lag_1']
        #Prediction.loc[Prediction.serd==serd,'Fatalities']=np.exp(Prediction.loc[Prediction.serd==serd,'Fatalities'])-1
    return(Prediction,test)        



Prediction,test=Postprocess(Feature_list,Num_lag,Hessian_Flag)
test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
test['Region']=test.Country_Region+test.Province_State.fillna('')
Prediction.loc[(Prediction.Date<=max(test.Date)) & (Prediction.Date>=min(test.Date)),'ForecastId']=test.loc[:,'ForecastId'].values
submission=Prediction.loc[Prediction.Date>=min(test.Date)][['ForecastId','ConfirmedCases','Fatalities']]
submission.ForecastId=submission.ForecastId.astype('int')
submission.sort_values('ForecastId',inplace=True)
submission.to_csv('submission2.csv',index=False)