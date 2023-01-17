# Import necessary libraries

import math

import pickle

import os

import pandas as pd

import folium 

import numpy as np

import matplotlib

matplotlib.use('nbagg')

import matplotlib.pylab as plt

import seaborn as sns

from matplotlib import rcParams

import plotly as py

import cufflinks

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objects as go

from tqdm import tqdm_notebook as tqdm

import warnings

import tensorflow as tf

from numpy import array

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import LSTM

from tensorflow.keras.utils import plot_model

from tensorflow.keras import Input

from tensorflow.keras.layers import BatchNormalization

from dateutil.relativedelta import relativedelta

import datetime

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Reading COVID-19 Raw data

# Reading COVID-19 Raw data

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

#covid_master=pd.read_csv('covid_19_data.csv')

submission = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

#covid_open=pd.read_csv('COVID19_open_line_list.csv')

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

#train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
# We will fill the missing states with a value 'NoState'

train=train.fillna('NoState')

test=test.fillna('NoState')

# changing the data type

train=train.rename(columns={'ConfirmedCases':'Confirmed','Fatalities':'Deaths','Country_Region':'Country/Region',

                     'Province_State':'Province/State','Date':'ObservationDate'})

test=test.rename(columns={'ConfirmedCases':'Confirmed','Fatalities':'Deaths','Country_Region':'Country/Region',

                     'Province_State':'Province/State','Date':'ObservationDate'})

num_cols=['Confirmed', 'Deaths']

for col in num_cols:

    temp=[int(i) for i in train[col]]

    train[col]=temp 

train.head(2)
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgbm

lb = LabelEncoder()

train_xgb=train.copy()

test_xgb=test.copy()

#lb.fit(train_xgb['Country/Region'])

train_xgb['Country/Region']=lb.fit_transform(train_xgb['Country/Region'])

train_xgb['Province/State']=lb.fit_transform(train_xgb['Province/State'])

test_xgb['Country/Region']=lb.fit_transform(test_xgb['Country/Region'])

test_xgb['Province/State']=lb.fit_transform(test_xgb['Province/State'])



train_dt=[int(datetime.datetime.strptime(train_xgb.iloc[i].ObservationDate, "%Y-%m-%d").strftime("%m%d")) 

          for i in range(len(train_xgb)) ]

train_xgb['ObservationDate']=train_dt

test_dt=[int(datetime.datetime.strptime(test_xgb.iloc[i].ObservationDate, "%Y-%m-%d").strftime("%m%d")) 

          for i in range(len(test_xgb)) ]

test_xgb['ObservationDate']=test_dt

train_xgb.head()
# Creating list of all regions of all counntries

unique_regions=train_xgb['Country/Region'].unique()

states_per_regions=[]

for reg in tqdm(unique_regions):

    states_per_regions.append(train_xgb[train_xgb['Country/Region']==reg]['Province/State'].unique()) 

print('No of unique regions:',len(unique_regions))    
train_xgb.head()
# Method for prediction

import math

def pred(model,data):

    y_pred=model.predict(data)

    #y_pred=[math.ceil(i) for i in y_pred]

    return y_pred
# Method for Hyperparameter Tuning

from sklearn.metrics import mean_squared_log_error

def get_best_xgb_model(X_c,y_c):

    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size=0.30, random_state=42)

    print('XGBoost Hyper Parameter Tunning')

    min_child_samples=[5,10,20,50,70]

    loss=[]

    loss1=[]

    loss2=[]

    loss3=[]

    loss4=[]

    for n in min_child_samples:

        xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=n)

        xgb_c.fit(X_train_c,y_train_c)

        y_pred=pred(xgb_c,X_val_c)

        if ((y_val_c >= 0).all() and (y_pred >= 0).all()):

            loss.append(mean_squared_log_error(y_pred,y_val_c))

            #print('min_child_samples:',n,'msle:',mean_squared_log_error(y_pred,y_val_c))

    print('Best min_child_samples:',min_child_samples[np.argmin(loss)])   



    learning_rate=[0.0001,0.001,0.01,0.1,0.2,0.5]  

    for n in learning_rate:

        xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=min_child_samples[np.argmin(loss)],learning_rate=n)

        xgb_c.fit(X_train_c,y_train_c)

        y_pred=pred(xgb_c,X_val_c)

        if ((y_val_c >= 0).all() and (y_pred >= 0).all()):

            loss1.append(mean_squared_log_error(y_pred,y_val_c))

        #print('learning_rate:',n,'msle:',mean_squared_log_error(y_pred,y_val_c))

    print('Best learning_rate:',learning_rate[np.argmin(loss1)])   



    num_leaves=[5,10,30,50,100]

    for n in num_leaves:

        xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=min_child_samples[np.argmin(loss)],learning_rate=learning_rate[np.argmin(loss1)]

                                ,num_leaves=n)

        xgb_c.fit(X_train_c,y_train_c)

        y_pred=pred(xgb_c,X_val_c)

        if ((y_val_c >= 0).all() and (y_pred >= 0).all()):

            loss2.append(mean_squared_log_error(y_pred,y_val_c))

        #print('num_leaves:',n,'msle:',mean_squared_log_error(y_pred,y_val_c))

    print('Best lnum_leaves:',num_leaves[np.argmin(loss2)])  



    reg_alpha=[0.0,0.01,0.05,0.1,0.5]

    for n in reg_alpha:

        xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=min_child_samples[np.argmin(loss)],learning_rate=learning_rate[np.argmin(loss1)]

                      ,reg_alpha=n,num_leaves=num_leaves[np.argmin(loss2)])

        xgb_c.fit(X_train_c,y_train_c)

        y_pred=pred(xgb_c,X_val_c)

        if ((y_val_c >= 0).all() and (y_pred >= 0).all()):

            loss3.append(mean_squared_log_error(y_pred,y_val_c))

        #print('reg_alpha:',n,'msle:',mean_squared_log_error(y_pred,y_val_c))

    print('Best reg_alpha:',reg_alpha[np.argmin(loss3)]) 



    n_estimators=[50,100,200,500,1000]

    for n in n_estimators:

        xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=min_child_samples[np.argmin(loss)],learning_rate=learning_rate[np.argmin(loss1)]

                      ,reg_alpha=reg_alpha[np.argmin(loss3)],num_leaves=num_leaves[np.argmin(loss2)],n_estimators=n)

        xgb_c.fit(X_train_c,y_train_c)

        y_pred=pred(xgb_c,X_val_c)

        if ((y_val_c >= 0).all() and (y_pred >= 0).all()):

            loss4.append(mean_squared_log_error(y_pred,y_val_c))

        #print('n_estimators:',n,'msle:',mean_squared_log_error(y_pred,y_val_c))

    print('Best n_estimators:',n_estimators[np.argmin(loss4)])   

    xgb_c=XGBRegressor(n_iterators=1000,min_child_samples=min_child_samples[np.argmin(loss)],learning_rate=learning_rate[np.argmin(loss1)]

                      ,reg_alpha=reg_alpha[np.argmin(loss3)],num_leaves=num_leaves[np.argmin(loss2)],n_estimators=n_estimators[np.argmin(loss4)])

    xgb_c.fit(X_train_c,y_train_c)

    return xgb_c
# Utility method to run Decision Tree model

from sklearn.tree import DecisionTreeClassifier

def run_model_DT(train,test):

    res=pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

    for k in tqdm(range(len(unique_regions))):

        for state in states_per_regions[k]:

            #print(unique_regions[k],state)

            temp_train=train[(train['Country/Region']==unique_regions[k]) &(train['Province/State']==state)]

            temp_test=test[(test['Country/Region']==unique_regions[k]) &(test['Province/State']==state)]

            X_train=temp_train.loc[:, ['Province/State', 'Country/Region','ObservationDate']]

            X_test=temp_test.loc[:, ['Province/State', 'Country/Region','ObservationDate']]

            y_c=temp_train.loc[:,'Confirmed']

            y_d=temp_train.loc[:,'Deaths']

            Forecast_Id=[int(i) for i in temp_test.ForecastId]

            # Model for Confirmed Cases

            #print(X_train.shape,len(y_c))

            model_c= DecisionTreeClassifier()

            model_c.fit(X_train, y_c)

            y_c_pred = model_c.predict(X_test)

            # Model for Confirmed Cases

            model_d= DecisionTreeClassifier()

            model_d.fit(X_train, y_d)

            y_d_pred = model_d.predict(X_test)

            res_temp=pd.DataFrame({'ForecastId': Forecast_Id, 'ConfirmedCases': y_c_pred, 'Fatalities': y_d_pred})

            res = pd.concat([res, res_temp], axis=0)

    return res



# Utility method to run XGBoost

def run_model_XGB(train,test,n_estimators):

    res=pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

    for k in tqdm(range(len(unique_regions))):

        for state in states_per_regions[k]:

            #print(unique_regions[k],state)

            temp_train=train[(train['Country/Region']==unique_regions[k]) &(train['Province/State']==state)]

            temp_test=test[(test['Country/Region']==unique_regions[k]) &(test['Province/State']==state)]

            X_train=temp_train.loc[:, ['Province/State', 'Country/Region','ObservationDate']]

            X_test=temp_test.loc[:, ['Province/State', 'Country/Region','ObservationDate']]

            y_c=temp_train.loc[:,'Confirmed']

            y_d=temp_train.loc[:,'Deaths']

            Forecast_Id=[int(i) for i in temp_test.ForecastId]

            # Model for Confirmed Cases

            #print(X_train.shape,len(y_c))

            model_c=XGBRegressor(n_estimators=n_estimators)

            model_c.fit(X_train, y_c)

            y_c_pred = model_c.predict(X_test)

            # Model for Death Cases

            model_d= XGBRegressor(n_estimators=n_estimators)

            model_d.fit(X_train, y_d)

            y_d_pred = model_d.predict(X_test)

            res_temp=pd.DataFrame({'ForecastId': Forecast_Id, 'ConfirmedCases': y_c_pred, 'Fatalities': y_d_pred})

            res = pd.concat([res, res_temp], axis=0)

    return res
# Run Model

res_DT=run_model_DT(train_xgb,test_xgb)

res_XGB=run_model_XGB(train_xgb,test_xgb,1500)
def get_mse(res,target):

    res=res.rename(columns={'ConfirmedCases':'Confirmed','Fatalities':'Deaths','Country_Region':'Country/Region',

                     'Province_State':'Province/State','Date':'ObservationDate'})

    Id=[int(i) for i in res.ForecastId]

    res['ForecastId']=Id

    temp=pd.merge(res,test,on='ForecastId',how='inner')

    y_pred=list(temp.query("ObservationDate>='2020-03-26' and ObservationDate<'2020-04-07'")[target])

    y_true=list(train.query("ObservationDate>='2020-03-26' and ObservationDate<'2020-04-07'")[target])

    print('mse:',mean_squared_error(y_true,y_pred))
res_DT.head()
res_final=res_XGB

#res_final['ConfirmedCases']=0.35*res_XGB['ConfirmedCases']+0.65*res_DT['ConfirmedCases']

#res_final['Fatalities']=0.35*res_XGB['Fatalities']+0.65*res_DT['Fatalities']

res_final.head()


Id=[int(i) for i in res_final.ForecastId]

res_final['ForecastId']=Id

res_final.to_csv('submission.csv',index=None)

res_final.head(20)