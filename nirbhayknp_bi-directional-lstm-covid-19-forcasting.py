import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
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
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link_1 = 'https://drive.google.com/open?id=1K98OeZKElrRsWmWjUnOBqunanSomqf-_'
fluff, id_1 = link_1.split('=')
downloaded = drive.CreateFile({'id':id_1}) 
downloaded.GetContentFile('train.csv')  
train = pd.read_csv('train.csv')
print (id_1)
link_2 = 'https://drive.google.com/open?id=1kGQN4KzoT1XGXDgZxVAQ-myYwSm5JThw'
fluff, id_2 = link_2.split('=')
downloaded = drive.CreateFile({'id':id_2}) 
downloaded.GetContentFile('test.csv')
test = pd.read_csv('test.csv')
print (id_2)
# 
link_3 = 'https://drive.google.com/open?id=1KAU9BiD5ALn0op_Fwvzps_w-6Cq21m7g'
fluff, id_3 = link_3.split('=')
downloaded = drive.CreateFile({'id':id_3}) 
downloaded.GetContentFile('submission.csv')
submission = pd.read_csv('submission.csv')
print (id_3)
train.isna().sum()
#filling the missing states with a value 'NoState'
train=train.fillna('NoState')
test=test.fillna('NoState')
#changing the data type
train=train.rename(columns={'ConfirmedCases':'Confirmed','Fatalities':'Deaths','Country_Region':'Country/Region',
                     'Province_State':'Province/State','Date':'ObservationDate'})
num_cols=['Confirmed', 'Deaths']
for col in num_cols:
    temp=[int(i) for i in train[col]]
    train[col]=temp 
train.head(2)
# list of all regions of all counntries
unique_regions=train['Country/Region'].unique()
states_per_regions=[]
for reg in tqdm(unique_regions):
    states_per_regions.append(train[train['Country/Region']==reg]['Province/State'].unique()) 
print('Num. of unique regions:',len(unique_regions)) 
def create_train_dataset(target,n_steps,train):
    x=[]
    y=[]
    for k in tqdm(range(len(unique_regions))):
        for state in states_per_regions[k]:
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
def create_test_dataset(target,n_steps,train):
    train = train.query("ObservationDate<'2020-03-26'")
    x=[]
    regs=[]
    for k in tqdm(range(len(unique_regions))):
        for state in states_per_regions[k]:
            temp=train[(train['Country/Region']==unique_regions[k]) &(train['Province/State']==state)]
            sequence=temp[target]
            x.append(sequence[len(sequence)-n_steps:len(sequence)+1])
            regs.append((unique_regions[k],state))      
    return array(x),regs
n_steps=7
print('Datasets with Positive Incidents..')
X_c,y_c=create_train_dataset('Confirmed',n_steps,train)
test_confirmed,regs= create_test_dataset('Confirmed',n_steps,train)
print('Datasets with Deaths Incidents..')
X_d,y_d=create_train_dataset('Deaths',n_steps,train)
test_deaths,regs= create_test_dataset('Deaths',n_steps,train)
print('Datasets prepared sucessfully.')
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size=0.20, random_state=42)
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=0.20, random_state=42)
X_train_c = X_train_c.reshape((X_train_c.shape[0], 1, X_train_c.shape[1]))
X_val_c= X_val_c.reshape(( X_val_c.shape[0], 1,  X_val_c.shape[1]))
X_test_c= test_confirmed.reshape(( test_confirmed.shape[0], 1, test_confirmed.shape[1]))
print(X_train_c.shape, y_train_c.shape, X_val_c.shape, y_val_c.shape,X_test_c.shape)
X_train_d = X_train_d.reshape((X_train_d.shape[0], 1, X_train_d.shape[1]))
X_val_d= X_val_d.reshape(( X_val_d.shape[0], 1,  X_val_d.shape[1]))
X_test_d= test_deaths.reshape(( test_deaths.shape[0], 1, test_deaths.shape[1]))
print(X_train_d.shape, y_train_d.shape, X_val_d.shape, y_val_d.shape,X_test_d.shape)
print(X_test_c[0])
print(X_test_d[0])
epochs = 30
batch_size = 32
n_hidden = 32
timesteps = X_train_c.shape[1]
input_dim = X_train_c.shape[2]
n_features=1

print(timesteps)
print(input_dim)
print(len(X_train_c))
model_c = Sequential()
model_c.add(LSTM(50, activation='relu', input_shape=(n_features,n_steps),return_sequences=True))
model_c.add(LSTM(150, activation='relu'))
model_c.add(Dense(1))
model_c.summary()
model_c.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
# fit the model
hist=model_c.fit(X_train_c,y_train_c, epochs=epochs, batch_size=batch_size, validation_data=(X_val_c, y_val_c), verbose=2, 
               shuffle=True,callbacks=callbacks)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Epoch vs Loss for Positive Cases')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
model_d = Sequential()
model_d.add(LSTM(50, activation='relu', input_shape=(n_features,n_steps),return_sequences=True))
model_d.add(LSTM(150, activation='relu'))
model_d.add(Dense(1))
model_d.summary()
model_d.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
# fit the model
hist=model_d.fit(X_train_d,y_train_d, epochs=epochs, batch_size=batch_size, validation_data=(X_val_d, y_val_d), verbose=2, 
               shuffle=True,callbacks=callbacks)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Epoch vs Loss for Death Cases')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
import math
def pred(model,data):
    y_pred=model.predict(data)
    y_pred=[math.ceil(i) for i in y_pred]
    return y_pred
def forcast(model,data,start_date,num_days):
    res_=dict()
    for i in range(len(data)):
        res_[i]=[]
    y_pred=pred(model,data)
    dates=[]
    date1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    for j in range(1,num_days+1):
        for i in range(len(data)):
            cur_window=list(data[i][0][1:8])
            #print(j,i,cur_window[-1])
            res_[i].append(cur_window[-1])
            cur_window.append(y_pred[i])
            data[i][0]=cur_window
        y_pred=pred(model,data)
        dates.append(date1.strftime("%Y-%m-%d"))
        date1+=relativedelta(days=1)
    res=pd.DataFrame(pd.DataFrame(pd.DataFrame(res_).values.T)) 
    res.columns=dates
    #print(res_)
    res['Country/State']=regs
    return res
def prepare_submission(res_c,res_d,test):
    index=dict()
    for i in range(len(res_c)):
        index[res_c.iloc[i]['Country/State']]=i
    pred_c=[]
    pred_d=[]
    for i in tqdm(range(len(test))):
        if((test.iloc[i]['Country_Region'],test.iloc[i]['Province_State']) in index):
            loc=index[(test.iloc[i]['Country_Region'],test.iloc[i]['Province_State'])]
            #print(res.iloc[loc][test.iloc[i]['Date']])
            pred_c.append(res_c.iloc[loc][test.iloc[i]['Date']])     
            pred_d.append(res_d.iloc[loc][test.iloc[i]['Date']]) 
    test['ConfirmedCases']=pred_c
    test['Fatalities']=pred_d
    res_regional=test
    res=test.drop(columns=['Province_State','Country_Region','Date'])
    return res,res_regional 
def get_countrywise_forcast(country_name,state_name,num_days):
    temp=train[(train['Country/Region']==country_name)&(train['Province/State']==state_name)].query("ObservationDate>'2020-03-25'")
    x_truth=temp.ObservationDate
    y_truth=temp.Confirmed
    pred_=res_regional[(res_regional['Country_Region']==country_name) & ((res_regional['Province_State']==state_name))]
    x_pred=pred_.Date[0:num_days]
    y_pred=pred_.ConfirmedCases[0:num_days]
    return list(x_truth),list(y_truth),list(x_pred),list(y_pred)
def get_countrywise_forcast_Deaths(country_name,state_name,num_days):
    temp=train[(train['Country/Region']==country_name)&(train['Province/State']==state_name)].query("ObservationDate>'2020-03-25'")
    x_truth=temp.ObservationDate
    y_truth=temp.Deaths
    pred_=res_regional[(res_regional['Country_Region']==country_name) & ((res_regional['Province_State']==state_name))]
    x_pred=pred_.Date[0:num_days]
    y_pred=pred_.Fatalities[0:num_days]
    return list(x_truth),list(y_truth),list(x_pred),list(y_pred)
res_confirmed=forcast(model_c,X_test_c,'2020-03-26',num_days=44)
res_deaths=forcast(model_d,X_test_d,'2020-03-26',num_days=44)
sub,res_regional=prepare_submission(res_confirmed,res_deaths,test)
sub.to_csv('submission.csv',index=None)
sub.head()
x_truth_Pa,y_truth_Pa,x_pred_Pa,y_pred_Pa=get_countrywise_forcast('Pakistan','NoState',15)
x_truth_In,y_truth_In,x_pred_In,y_pred_In=get_countrywise_forcast('India','NoState',15)
x_truth_Uk,y_truth_Uk,x_pred_Uk,y_pred_Uk=get_countrywise_forcast('United Kingdom','NoState',15)
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

fig.add_trace(go.Scatter(x=x_truth_Uk, 
                         y=y_truth_Uk,
                         mode='lines+markers',
                         name='Actual_UK',
                         line=dict(color='yellow', width=3)),1,2)
fig.add_trace(go.Scatter(x=x_pred_Uk, 
                         y=y_pred_Uk,
                         mode='lines+markers',
                         name='Predicted_Uk',
                         line=dict(color='red', width=1)),1,2)

fig.add_trace(go.Scatter(x=x_truth_Pa, 
                         y=y_truth_Pa,
                         mode='lines+markers',
                         name='Actual_Pakistan',
                         line=dict(color='#E5CCFF', width=3)),2,1)
fig.add_trace(go.Scatter(x=x_pred_Pa, 
                         y=y_pred_Pa,
                         mode='lines+markers',
                         name='Predicted_Pakistan',
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
                  title = 'COVID-19 Confirmed Cases prediction in India/Pakistan/United Kingdom/Italy(27th March - 9th April)',
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

fig.add_trace(go.Scatter(x=x_truth_Uk, 
                         y=y_truth_Uk,
                         mode='lines+markers',
                         name='Actual_UK',
                         line=dict(color='yellow', width=3)),1,2)
fig.add_trace(go.Scatter(x=x_pred_Uk, 
                         y=y_pred_Uk,
                         mode='lines+markers',
                         name='Predicted_UK',
                         line=dict(color='red', width=1)),1,2)

fig.add_trace(go.Scatter(x=x_truth_Pa, 
                         y=y_truth_Pa,
                         mode='lines+markers',
                         name='Actual_Pakitan',
                         line=dict(color='#E5CCFF', width=3)),2,1)
fig.add_trace(go.Scatter(x=x_pred_Pa, 
                         y=y_pred_Pa,
                         mode='lines+markers',
                         name='Predicted_Pakistan',
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
                  title = 'COVID-19 Death Cases prediction in India/Pakistan/United Kingdom/Italy(27th March - 9th April)',
                  annotations=[
    ]
                 )

