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

warnings.filterwarnings("ignore")

# Reading COVID-19 Raw data

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

#covid_master=pd.read_csv('covid_19_data.csv')

submission = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

#covid_open=pd.read_csv('COVID19_open_line_list.csv')

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

#train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
train.isna().sum()
# We will fill the missing states with a value 'NoState'

train=train.fillna('NoState')

test=test.fillna('NoState')

# changing the data type

train=train.rename(columns={'ConfirmedCases':'Confirmed','Fatalities':'Deaths','Country_Region':'Country/Region',

                     'Province_State':'Province/State','Date':'ObservationDate'})

num_cols=['Confirmed', 'Deaths']

for col in num_cols:

    temp=[int(i) for i in train[col]]

    train[col]=temp 

train.head(2)
# Creating list of all regions of all counntries

unique_regions=train['Country/Region'].unique()

states_per_regions=[]

for reg in tqdm(unique_regions):

    states_per_regions.append(train[train['Country/Region']==reg]['Province/State'].unique()) 

print('No of unique regions:',len(unique_regions))    
# function to create training data for LSTM

# We will take last 7 days Cases as input and 8th day's case as output

def create_train_dataset(target,n_steps,train,pivot_date):

    train = train.query("ObservationDate<"+pivot_date)

    x=[]

    y=[]

    for k in tqdm(range(len(unique_regions))):

        for state in states_per_regions[k]:

            #print(unique_regions[k],state)

            temp=train[(train['Country/Region']==unique_regions[k]) &(train['Province/State']==state)]

            sequence=list(temp[target])

            for i in range(len(sequence)):

                end_ix = i + n_steps

                if end_ix > len(sequence)-1:

                    break

                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

                if(seq_y!=0):

                    x.append(seq_x)   

                    y.append(seq_y)

    return array(x),array(y)    



def create_countrywise_newly_added_train_dataset(target,n_steps,train,pivot_date):

    train = train.query("ObservationDate<"+pivot_date)

    x=[]

    y=[]

    for k in tqdm(range(len(unique_regions))):

            #print(unique_regions[k],state)

        temp=train[(train['Country/Region']==unique_regions[k])]

        sequence=list(temp[target])

        for i in range(len(sequence)):

            end_ix = i + n_steps

            if end_ix > len(sequence)-1:

                break

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

            if(seq_y!=0):

                x.append(seq_x)   

                y.append(seq_y)

    return array(x),array(y)  
# function to create test dataset

# our supervised probem is now given last 7 days data predict the no of cases for 8th day

# target : 'Confirmed'/'Deaths'

def create_test_dataset(target,n_steps,train,pivot_date):

    train = train.query("ObservationDate<"+pivot_date)

    x=[]

    regs=[]

    for k in tqdm(range(len(unique_regions))):

        for state in states_per_regions[k]:

            #regs.append((unique_regions[k],state))

            temp=train[(train['Country/Region']==unique_regions[k]) &(train['Province/State']==state)]

            sequence=temp[target]

            #print(sequence[len(sequence)-n_steps:len(sequence)+1])

            x.append(sequence[len(sequence)-n_steps:len(sequence)+1])

            regs.append((unique_regions[k],state))      

    return array(x),regs



def create_countrywise_newly_added_test_dataset(target,n_steps,train,pivot_date):

    train = train.query("ObservationDate<"+pivot_date)

    x=[]

    regs=[]

    for k in tqdm(range(len(unique_regions))):

        temp=train[(train['Country/Region']==unique_regions[k])]

        sequence=temp[target]

        #print(sequence[len(sequence)-n_steps:len(sequence)+1])

        x.append(sequence[len(sequence)-n_steps:len(sequence)+1])

        regs.append(unique_regions[k])      

    return array(x),regs

def get_newly_added(world_data_):

    world_data_=world_data_.sort_values(['Country/Region','ObservationDate'])

    temp=[0*i for i in range(len(world_data_))]

    world_data_['New Confirmed']=temp

    world_data_['New Death']=temp

    for i in tqdm(range(1,len(world_data_))):

        if(world_data_['Country/Region'].iloc[i]==world_data_['Country/Region'].iloc[i-1]):

            if(world_data_['Deaths'].iloc[i]<world_data_['Deaths'].iloc[i-1]):

                world_data_['Deaths'].iloc[i]=world_data_['Deaths'].iloc[i-1]

            if(world_data_['Confirmed'].iloc[i]<world_data_['Confirmed'].iloc[i-1]):

                world_data_['Confirmed'].iloc[i]=world_data_['Confirmed'].iloc[i-1]     

            world_data_['New Confirmed'].iloc[i]=world_data_['Confirmed'].iloc[i]-world_data_['Confirmed'].iloc[i-1]

            world_data_['New Death'].iloc[i]=world_data_['Deaths'].iloc[i]-world_data_['Deaths'].iloc[i-1]

        else:

            world_data_['New Confirmed'].iloc[i]=world_data_['Confirmed'].iloc[i]

            world_data_['New Death'].iloc[i]=world_data_['Deaths'].iloc[i]

    return world_data_
# Countrywise timeseries data with Newly added Incident Each Day

covid_timeseries = train.groupby(['ObservationDate','Country/Region','Province/State'])['Confirmed', 'Deaths'].sum()

covid_timeseries=covid_timeseries.reset_index().sort_values('ObservationDate')

covid_timeseries=get_newly_added(covid_timeseries)

covid_timeseries[covid_timeseries['Country/Region']=='India'].tail()
# Maintain the date format for pivot_date and forcast_start_date

# Pivot_date : data of date less than the given date will be used for training

# Forcast_start_date : Date from which forcasting will be started

n_steps=7

pivot_date="'2020-04-02'"

forcast_start_date='2020-04-02'

print('Preparing datasets with Cumulative Confirmed Incidents..')

X_c,y_c=create_train_dataset('Confirmed',n_steps,train,pivot_date)

print('Preparing datasets with Newly Confirmed Incidents..')

X_nc,y_nc=create_train_dataset('New Confirmed',n_steps,covid_timeseries,pivot_date)

test_confirmed,regs= create_test_dataset('Confirmed',n_steps,train,pivot_date)

test_nc,reg_nc=create_test_dataset('New Confirmed',n_steps,covid_timeseries,pivot_date)

print('Preparing datasets with Deaths Incidents..')

X_d,y_d=create_train_dataset('Deaths',n_steps,train,pivot_date)

test_deaths,regs= create_test_dataset('Deaths',n_steps,train,pivot_date)

print('Datasets prepared sucessfully.')
# Split the train data in to train and val data



X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size=0.30, random_state=42)

X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=0.30, random_state=42)



X_train_nc, X_val_nc, y_train_nc, y_val_nc = train_test_split(X_c, y_c, test_size=0.30, random_state=42)

# Reshapping the Confirmed data for LSTM

X_train_c = X_train_c.reshape((X_train_c.shape[0], 1, X_train_c.shape[1]))

X_val_c= X_val_c.reshape(( X_val_c.shape[0], 1,  X_val_c.shape[1]))

X_train_nc = X_train_nc.reshape((X_train_nc.shape[0], 1, X_train_nc.shape[1]))

X_val_nc= X_val_nc.reshape(( X_val_nc.shape[0], 1,  X_val_nc.shape[1]))

X_test_c= test_confirmed.reshape(( test_confirmed.shape[0], 1, test_confirmed.shape[1]))

X_test_nc= test_nc.reshape(( test_nc.shape[0], 1, test_nc.shape[1]))

print(X_train_c.shape, y_train_c.shape, X_val_c.shape, y_val_c.shape,X_test_c.shape,X_test_nc.shape)
# Reshapping the donfirmed data for LSTM

X_train_d = X_train_d.reshape((X_train_d.shape[0], 1, X_train_d.shape[1]))

X_val_d= X_val_d.reshape(( X_val_d.shape[0], 1,  X_val_d.shape[1]))

X_test_d= test_deaths.reshape(( test_deaths.shape[0], 1, test_deaths.shape[1]))

print(X_train_d.shape, y_train_d.shape, X_val_d.shape, y_val_d.shape,X_test_d.shape)
print(X_train_c[100])

print(X_train_d[1])
# Initializing model components

epochs = 10

batch_size = 32

n_hidden = 32

timesteps = X_train_c.shape[1]

input_dim = X_train_c.shape[2]

n_features=1



print(timesteps)

print(input_dim)

print(len(X_train_c))
# Stacked LSTM Model

model_c = Sequential()

model_c.add(LSTM(50, activation='relu', input_shape=(n_features,n_steps),return_sequences=True))

model_c.add(LSTM(150, activation='relu'))

model_c.add(Dense(1,activation='relu'))

model_c.summary()
# Compiling the model

model_c.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=20),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# fit the model

hist=model_c.fit(X_train_c,y_train_c, epochs=epochs, batch_size=batch_size, validation_data=(X_val_c, y_val_c), verbose=2, 

               shuffle=True,callbacks=callbacks)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Epoch vs Loss for Confirmed Cases')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
# Stadked LSTM Model

model_d = Sequential()

model_d.add(LSTM(50, activation='relu', input_shape=(n_features,n_steps),return_sequences=True))

model_d.add(LSTM(50, activation='relu'))

model_d.add(Dense(1))

model_d.summary()
# Compiling the model

model_d.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=20),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# fit the model

hist=model_d.fit(X_train_d,y_train_d, epochs=epochs, batch_size=batch_size, validation_data=(X_val_d, y_val_d), verbose=2, 

               shuffle=True,callbacks=callbacks)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Epoch vs Loss for Deathh Cases')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
# Stadked LSTM Model

model_nc = Sequential()

model_nc.add(LSTM(50, activation='relu', input_shape=(n_features,n_steps),return_sequences=True))

model_nc.add(LSTM(50, activation='relu'))

model_nc.add(Dense(1))

model_nc.summary()

model_nc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=20),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# fit the model

hist=model_nc.fit(X_train_nc,y_train_nc, epochs=epochs, batch_size=batch_size, validation_data=(X_val_nc, y_val_nc), verbose=2, 

               shuffle=True,callbacks=callbacks)
import math

def pred(model,data):

    y_pred=model.predict(data)

    #y_pred=[math.ceil(i) for i in y_pred]

    return y_pred
# Utility method for Forcasting

# model - trained model on Confirmed/Deaths data

# start_date - Starting date of forcasting

# num_days - Number of days for which forcasting is required

def forcast(model,data,start_date,num_days):

    res_=dict()

    for i in range(len(data)):

        res_[i]=[]

    y_pred=pred(model,data)

    dates=[]

    date1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    for j in range(1,num_days+1):

        for i in range(len(data)):

            cur_window=list(data[i][0][1:n_steps+1])

            #print(j,i,cur_window[-1])

            res_[i].append(cur_window[-1])

            cur_window.append(y_pred[i])

            data[i][0]=cur_window

        y_pred=pred(model,data)

        dates.append(date1.strftime("%Y-%m-%d"))

        date1+=relativedelta(days=1)

    res=pd.DataFrame(pd.DataFrame(pd.DataFrame(res_).values.T)) 

    res.columns=dates

    res['Country/State']=regs

    return res



def forcast_(model,data,start_date,num_days):

    res_=[]

    for i in list(data['Country/Region']):

        res_.append(i)

    y_pred=pred(model,data)

    dates=[]

    date1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    for j in range(1,num_days+1):

        for i in range(len(data)):

            cur_window=list(data[i][0][1:n_steps+1])

            #print(j,i,cur_window[-1])

            res_[i].append(cur_window[-1])

            cur_window.append(y_pred[i])

            data[i][0]=cur_window

        y_pred=pred(model,data)

        dates.append(date1.strftime("%Y-%m-%d"))

        date1+=relativedelta(days=1)

    res=pd.DataFrame(pd.DataFrame(pd.DataFrame(res_).values.T)) 

    res.columns=dates

    res['Country/State']=res_

    return res

        
# Utility method for submission

def prepare_submission(res_c,res_d,res_nc,test,pivot_date):

    test=test.query("Date>="+pivot_date)

    index=dict()

    for i in range(len(res_c)):

        index[res_c.iloc[i]['Country/State']]=i

    pred_c=[]

    pred_d=[]

    pred_nc=[]

    for i in tqdm(range(len(test))):

        if((test.iloc[i]['Country_Region'],test.iloc[i]['Province_State']) in index):

            loc=index[(test.iloc[i]['Country_Region'],test.iloc[i]['Province_State'])]

            #print(res.iloc[loc][test.iloc[i]['Date']])

            pred_c.append(res_c.iloc[loc][test.iloc[i]['Date']])     

            pred_d.append(res_d.iloc[loc][test.iloc[i]['Date']]) 

            pred_nc.append(res_nc.iloc[loc][test.iloc[i]['Date']]) 

    test['ConfirmedCases']=pred_c

    test['Fatalities']=pred_d

    test['New Confirmed']=pred_nc

    res_regional=test

    res=test.drop(columns=['Province_State','Country_Region','Date','New Confirmed'])

    return res,res_regional 
# Call only when forcast and submission data are available

def get_countrywise_forcast_(target,country_name,state_name,num_days):

    temp=covid_timeseries[(covid_timeseries['Country/Region']==country_name)&(covid_timeseries['Province/State']==state_name)].query("ObservationDate>="+pivot_date)

    x_truth=temp.ObservationDate

    y_truth=temp[target]

    pred_=res_regional[(res_regional['Country_Region']==country_name) & ((res_regional['Province_State']==state_name))]

    x_pred=pred_.Date[0:num_days]

    y_pred=pred_[target][0:num_days]

    return list(x_truth),list(y_truth),list(x_pred),list(y_pred)
# Call only when forcast and submission data are available

def get_countrywise_forcast(country_name,state_name,num_days):

    temp=train[(train['Country/Region']==country_name)&(train['Province/State']==state_name)].query("ObservationDate>="+pivot_date)

    x_truth=temp.ObservationDate

    y_truth=temp.Confirmed

    pred_=res_regional[(res_regional['Country_Region']==country_name) & ((res_regional['Province_State']==state_name))]

    x_pred=pred_.Date[0:num_days]

    y_pred=pred_.ConfirmedCases[0:num_days]

    return list(x_truth),list(y_truth),list(x_pred),list(y_pred)
# Call only when forcast and submission data are available

def get_countrywise_forcast_Deaths(country_name,state_name,num_days):

    temp=train[(train['Country/Region']==country_name)&(train['Province/State']==state_name)].query("ObservationDate>="+pivot_date)

    x_truth=temp.ObservationDate

    y_truth=temp.Deaths

    pred_=res_regional[(res_regional['Country_Region']==country_name) & ((res_regional['Province_State']==state_name))]

    x_pred=pred_.Date[0:num_days]

    y_pred=pred_.Fatalities[0:num_days]

    return list(x_truth),list(y_truth),list(x_pred),list(y_pred)
# Utility Method to convert newly added prediction to cumulative [Not Accurate]

def get_cumulative_confirmed_cases(world_data_):

    world_data_=world_data_.sort_values(['Country_Region','Date'])

    temp=[0*i for i in range(len(world_data_))]

    world_data_['Cumulative Confirmed']=world_data_['New Confirmed']

    for i in tqdm(range(1,len(world_data_))):

        if(world_data_['Country_Region'].iloc[i]!=world_data_['Country_Region'].iloc[i-1]):

            world_data_['Cumulative Confirmed'].iloc[i]=world_data_['ConfirmedCases'].iloc[i]

    for i in tqdm(range(1,len(world_data_))):    

        if(world_data_['Country_Region'].iloc[i]==world_data_['Country_Region'].iloc[i-1]):

            world_data_['Cumulative Confirmed'].iloc[i]=world_data_['Cumulative Confirmed'].iloc[i]+world_data_['Cumulative Confirmed'].iloc[i-1]

    return world_data_
# num_days = Num of days for which Forcasting is required

#forcast_start_date='2020-04-01'



res_confirmed=forcast(model_c,X_test_c,forcast_start_date,num_days=50)

res_deaths=forcast(model_d,X_test_d,forcast_start_date,num_days=50)

res_new_confirmed=forcast(model_nc,X_test_nc,forcast_start_date,num_days=50)
# res_regional contains submission data along with extra columns

sub,res_regional=prepare_submission(res_confirmed,res_deaths,res_new_confirmed,test,pivot_date)

sub.to_csv('submission.csv',index=None)

sub.head()
x_truth_Ge,y_truth_Ge,x_pred_Ge,y_pred_Ge=get_countrywise_forcast_('New Confirmed','Germany','NoState',15)

x_truth_In,y_truth_In,x_pred_In,y_pred_In=get_countrywise_forcast_('New Confirmed','India','NoState',15)

x_truth_Sp,y_truth_Sp,x_pred_Sp,y_pred_Sp=get_countrywise_forcast_('New Confirmed','Spain','NoState',15)

x_truth_It,y_truth_It,x_pred_It,y_pred_It=get_countrywise_forcast_('New Confirmed','Italy','NoState',15)
x_truth_Ge,y_truth_Ge,x_pred_Ge,y_pred_Ge=get_countrywise_forcast('Germany','NoState',15)

x_truth_In,y_truth_In,x_pred_In,y_pred_In=get_countrywise_forcast('India','NoState',15)

x_truth_Sp,y_truth_Sp,x_pred_Sp,y_pred_Sp=get_countrywise_forcast('Spain','NoState',15)

x_truth_It,y_truth_It,x_pred_It,y_pred_It=get_countrywise_forcast('Italy','NoState',15)
fig = make_subplots(rows=2, cols=2)

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=x_truth_In, 

                         y=y_truth_In,

                         mode='lines+markers',

                         name='Actual_India',

                         line=dict(color='#CCFFCC', width=3)),1,1)

fig.add_trace(go.Scatter(x=x_pred_In, 

                         y=y_pred_In,

                         mode='lines+markers',

                         name='Predicted_India',

                         line=dict(color='red', width=1)),1,1)



fig.add_trace(go.Scatter(x=x_truth_Sp, 

                         y=y_truth_Sp,

                         mode='lines+markers',

                         name='Actual_Spain',

                         line=dict(color='yellow', width=3)),1,2)

fig.add_trace(go.Scatter(x=x_pred_Sp, 

                         y=y_pred_Sp,

                         mode='lines+markers',

                         name='Predicted_Spain',

                         line=dict(color='red', width=1)),1,2)



fig.add_trace(go.Scatter(x=x_truth_Ge, 

                         y=y_truth_Ge,

                         mode='lines+markers',

                         name='Actual_Germany',

                         line=dict(color='#E5CCFF', width=3)),2,1)

fig.add_trace(go.Scatter(x=x_pred_Ge, 

                         y=y_pred_Ge,

                         mode='lines+markers',

                         name='Predicted_Germany',

                         line=dict(color='red', width=1)),2,1)





fig.add_trace(go.Scatter(x=x_truth_It, 

                         y=y_truth_It,

                         mode='lines+markers',

                         name='Actual-Italy',

                         line=dict(color='#33FFFF', width=3)),2,2)

fig.add_trace(go.Scatter(x=x_pred_It, 

                         y=y_pred_It,

                         mode='lines+markers',

                         name='Predicted_Italy',

                         line=dict(color='red', width=1)),2,2)



fig.update_layout(template='plotly_dark',

                  title = 'COVID-19 Confirmed Cases prediction in India/Spain/Germany/Italy(27th March - 9th April)',

                  annotations=[

    ]

                 )
x_truth_Ge,y_truth_Ge,x_pred_Ge,y_pred_Ge=get_countrywise_forcast_Deaths('Germany','NoState',15)

x_truth_In,y_truth_In,x_pred_In,y_pred_In=get_countrywise_forcast_Deaths('India','NoState',15)

x_truth_Sp,y_truth_Sp,x_pred_Sp,y_pred_Sp=get_countrywise_forcast_Deaths('Spain','NoState',15)

x_truth_It,y_truth_It,x_pred_It,y_pred_It=get_countrywise_forcast_Deaths('Italy','NoState',15)
fig = make_subplots(rows=2, cols=2)

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=x_truth_In, 

                         y=y_truth_In,

                         mode='lines+markers',

                         name='Actual_India',

                         line=dict(color='#CCFFCC', width=3)),1,1)

fig.add_trace(go.Scatter(x=x_pred_In, 

                         y=y_pred_In,

                         mode='lines+markers',

                         name='Predicted_India',

                         line=dict(color='red', width=1)),1,1)



fig.add_trace(go.Scatter(x=x_truth_Sp, 

                         y=y_truth_Sp,

                         mode='lines+markers',

                         name='Actual_Spain',

                         line=dict(color='yellow', width=3)),1,2)

fig.add_trace(go.Scatter(x=x_pred_Sp, 

                         y=y_pred_Sp,

                         mode='lines+markers',

                         name='Predicted_Spain',

                         line=dict(color='red', width=1)),1,2)



fig.add_trace(go.Scatter(x=x_truth_Ge, 

                         y=y_truth_Ge,

                         mode='lines+markers',

                         name='Actual_Germany',

                         line=dict(color='#E5CCFF', width=3)),2,1)

fig.add_trace(go.Scatter(x=x_pred_Ge, 

                         y=y_pred_Ge,

                         mode='lines+markers',

                         name='Predicted_Germany',

                         line=dict(color='red', width=1)),2,1)





fig.add_trace(go.Scatter(x=x_truth_It, 

                         y=y_truth_It,

                         mode='lines+markers',

                         name='Actual-Italy',

                         line=dict(color='#33FFFF', width=3)),2,2)

fig.add_trace(go.Scatter(x=x_pred_It, 

                         y=y_pred_It,

                         mode='lines+markers',

                         name='Predicted_Italy',

                         line=dict(color='red', width=1)),2,2)



fig.update_layout(template='plotly_dark',

                  title = 'COVID-19 Death Cases prediction in India/Spain/Germany/Italy(27th March - 9th April)',

                  annotations=[

    ]

                 )