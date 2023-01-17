# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/ece657aw20asg4coronavirus'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


confirmed_set = pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
death_set = pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
recovered_set = pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
from datetime import datetime
plt.style.use('seaborn')
%matplotlib inline 
confirmed_set.head(3)
death_set.head(3)
recovered_set.head(3)
COUNTRY = 'India'
confirmed_set_t = confirmed_set.melt(id_vars = ["Country/Region", "Province/State","Lat","Long"],var_name = "Date",value_name="Confirmed")
confirmed_set_t.head(5)
death_set_t = death_set.melt(id_vars = ["Country/Region", "Province/State","Lat","Long"],var_name = "Date",value_name="death")
death_set_t.head(5)
recovered_set_t = recovered_set.melt(id_vars = ["Country/Region", "Province/State","Lat","Long"],var_name = "Date",value_name="recovered")
recovered_set_t.head(5)
if(COUNTRY=="World"):
    confirmed_set_t_India = confirmed_set_t.drop(["Country/Region","Province/State","Lat","Long"], axis=1)
    confirmed_set_t_India = confirmed_set_t_India.groupby(["Date"]).sum()

else:
    confirmed_set_t_India = confirmed_set_t[confirmed_set_t['Country/Region']==COUNTRY]
    confirmed_set_t_India = confirmed_set_t_India.drop(["Province/State","Lat","Long"], axis=1)
    confirmed_set_t_India = confirmed_set_t_India.groupby(["Country/Region","Date"]).sum()

confirmed_set_t_India = confirmed_set_t_India.reset_index()
confirmed_set_t_India["Date"] = pd.to_datetime(confirmed_set_t_India['Date'])
confirmed_set_t_India = confirmed_set_t_India.sort_values(by=["Date"])
confirmed_set_t_India.tail()
confirmed_set_t_India['frequency'] = confirmed_set_t_India[['Confirmed']].diff().fillna(confirmed_set_t_India)
confirmed_set_t_India.tail()
confirmed_set_t_India.shape
confirmed_set_t_India["Date"] = pd.to_datetime(confirmed_set_t_India['Date'])
confirmed_set_t_India_ts = confirmed_set_t_India.iloc[:,-3:]
confirmed_set_t_India_ts = confirmed_set_t_India_ts.set_index('Date')
confirmed_set_t_India_ts = confirmed_set_t_India_ts.drop(['Confirmed'], axis=1)
plt.plot(confirmed_set_t_India_ts)
plt.show()
Original = confirmed_set_t_India_ts
rolmean = confirmed_set_t_India_ts.rolling(50).mean()
rolstd = confirmed_set_t_India_ts.rolling(14).std()
rolmean7 = confirmed_set_t_India_ts.rolling(20).mean()
plt.figure(figsize=(20,10))
plt.plot(confirmed_set_t_India_ts, color='red',label='Original-India')
plt.plot(rolmean, color='blue', label='Rolling Mean-India')
plt.plot(rolmean7, color='green', label='Rolling Mean-India')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation-India')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(Original)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(Original, label="Original")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label="Seasonal")
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label="Residual")
plt.legend(loc='best')
from fbprophet import Prophet
cap = 300   #china 
floor = 0
df = residual.reset_index()
df.columns = ['ds', 'y']
m_residual = Prophet()
m_residual.fit(df)
future_residual = m_residual.make_future_dataframe(periods=60)
future_residual = m_residual.predict(future_residual)
forcasted_residual = future_residual[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forcasted_residual.ds = pd.to_datetime(forcasted_residual.ds)
forcasted_residual = forcasted_residual.set_index('ds')
plt.figure(figsize=(50,10))
plt.plot(forcasted_residual.index, forcasted_residual.yhat)

df = trend.reset_index()
df.columns = ['ds', 'y']
df['cap'] = cap
df['floor'] = floor
m_trend = Prophet(growth='logistic')
m_trend.fit(df)
future_trend = m_trend.make_future_dataframe(periods=60)
future_trend['cap'] = cap
future_trend['floor'] = floor
future_trend = m_trend.predict(future_trend)
forcasted_trend = future_trend[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forcasted_trend.ds = pd.to_datetime(forcasted_trend.ds)
forcasted_trend = forcasted_trend.set_index('ds')
plt.figure(figsize=(50,10))
plt.plot(forcasted_trend.index, forcasted_trend.yhat)
df = seasonal.reset_index()
df.columns = ['ds', 'y']
m_seasonal = Prophet()
m_seasonal.fit(df)
future_seasonal = m_seasonal.make_future_dataframe(periods=60)
future_seasonal = m_seasonal.predict(future_seasonal)
forcasted_seasonal = future_seasonal[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forcasted_seasonal.ds = pd.to_datetime(forcasted_seasonal.ds)
forcasted_seasonal = forcasted_seasonal.set_index('ds')
plt.figure(figsize=(50,10))
plt.plot(forcasted_seasonal.index, forcasted_seasonal.yhat)
x1 = forcasted_residual.merge(forcasted_trend, on='ds')
x2 = x1.merge(forcasted_seasonal, on='ds')
forcasted_original = pd.DataFrame()
forcasted_original["predicted_cases"] = x2[['yhat_x','yhat_y','yhat']].sum(axis=1)
forcasted_original["min"] = x2[['yhat_lower_x','yhat_lower_y','yhat_lower']].sum(axis=1)
forcasted_original["max"] = x2[['yhat_upper_x','yhat_upper_y','yhat_upper']].sum(axis=1)
forcasted_original.tail()
import datetime
today = datetime.date.today()
today = today.strftime("%Y-%m-%d")
forcasted_original.loc[forcasted_original.index==today]


confirmed=confirmed_set_t.drop(['Country/Region', 'Province/State','Lat','Long'], axis=1)
confirmed=confirmed.groupby(['Date']).sum()
death=death_set_t.drop(['Country/Region', 'Province/State','Lat','Long'], axis=1)
death=death.groupby(['Date']).sum()
recovered=recovered_set_t.drop(['Country/Region', 'Province/State','Lat','Long'], axis=1)
recovered=recovered.groupby(['Date']).sum()
print(confirmed.shape, death.shape, recovered.shape)
dataf =  confirmed.merge(death,on='Date').merge(recovered,on='Date')
dataf.reset_index(inplace=True)
print("Check if any missing information in data:", dataf.isnull().values.any())
print(dataf.shape)
data=dataf
day = data["Date"].values
day = [my_str.split("/")[1] for my_str in day]
data["day"] = day
data.head()
del data['Date']
data.head()
dataf = data[["day", "Confirmed", "death", "recovered"]]
dataf.head(5)
dataf.shape
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler


from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape
from keras.callbacks import ModelCheckpoint
def load_data(data, time_step=2, after_day=1, validate_percent=0.67):
    seq_length = time_step + after_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])
    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]
def base_model(feature_len=3, after_day=3, input_shape=(8, 1)):
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))

    model.add(RepeatVector(after_day))
    model.add(LSTM(200, return_sequences=True))

    model.add(TimeDistributed(Dense(units=feature_len, activation='linear')))

    return model
def seq2seq(feature_len=1, after_day=1, input_shape=(8, 1)):
    '''
    Encoder:
    X = Input sequence
    C = LSTM(X); The context vector

    Decoder:
    y(t) = LSTM(s(t-1), y(t-1)); where s is the hidden state of the LSTM(h and c)
    y(0) = LSTM(s0, C); C is the context vector from the encoder.
    '''

    # Encoder
    encoder_inputs = Input(shape=input_shape) 
    encoder = LSTM(units=100, return_state=True,  name='encoder')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Decoder
    reshapor = Reshape((1, 100), name='reshapor')
    decoder = LSTM(units=100, return_sequences=True, return_state=True, name='decoder')

    # Densor
    densor_output = Dense(units=feature_len, activation='linear', name='output')

    inputs = reshapor(encoder_outputs)
    #inputs = tdensor(inputs)
    all_outputs = []



    for _ in range(after_day):
        outputs, h, c = decoder(inputs, initial_state=states)

        inputs = outputs
        states = [state_h, state_c]

        outputs = densor_output(outputs)
        all_outputs.append(outputs)

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)

    return model
def normalize_data(data, scaler, feature_len):
    minmaxscaler = scaler.fit(data)
    normalize_data = minmaxscaler.transform(data)
    return normalize_data

scaler = MinMaxScaler(feature_range=(0, 1))
data = normalize_data(dataf, scaler,dataf.shape[1])

x_train, y_train, x_validate, y_validate = load_data(data,time_step=3, after_day=4, validate_percent=0.50)
print('train data: ', x_train.shape, y_train.shape)
print('validate data: ', x_validate.shape, y_validate.shape)
input_shape = (3, data.shape[1])
model = seq2seq(data.shape[1], 4, input_shape)
model.compile(loss='mse', optimizer='adam',metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, batch_size=3, epochs=70)

import math 
print('-' * 100)
train_score = model.evaluate(x=x_train, y=y_train, batch_size=3, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE ) , %.8f  ACC' % (train_score[0], math.sqrt(train_score[0]),train_score[1]*100)  )
validate_score = model.evaluate(x=x_validate, y=y_validate, batch_size=3, verbose=0)
print('Validation Score: %.8f MSE (%.8f RMSE ) , %.8f  ACC' % (validate_score[0], math.sqrt(validate_score[0]),validate_score[1]*100))
train_predict = model.predict(x_train)
validate_predict = model.predict(x_validate)
def inverse_normalize_data(data, scaler):
    for i in range(len(data)):
        data[i] = scaler.inverse_transform(data[i])

    return data
train_predict = inverse_normalize_data(train_predict, scaler)
y_train = inverse_normalize_data(y_train, scaler)
validate_predict = inverse_normalize_data(validate_predict, scaler)
y_validate = inverse_normalize_data(y_validate, scaler)
day = ['First Day','Second Day','Third Day','Fourth Day']

fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)

ax1.plot(day,y_validate[:,:,1][3],color='red',label='Confirmed Actual')
ax1.plot(day,validate_predict[:,:,1][3],color='maroon',label='Confirmed Prediction')
ax1.title.set_text("Confirmed per/day")
ax1.legend()


ax2.bar(day,y_validate[:,:,2][3],color='red',label='Deaths Actual')
ax2.bar(day,validate_predict[:,:,2][3],color='maroon',label='Deaths Prediction')
ax2.title.set_text("Deaths per/day")
ax2.legend()


ax3.bar(day,y_validate[:,:,3][3],color='red',label='Recoverd Actual')
ax3.bar(day,validate_predict[:,:,3][3],color='maroon',label='Recoverd Prediction')
ax3.title.set_text("Recoverd per/day")
ax3.legend()

plt.show()
x_train, y_train, x_validate, y_validate = load_data(data,time_step=3, after_day=4, validate_percent=0.)

train_predict = inverse_normalize_data(train_predict, scaler)
y_train = inverse_normalize_data(y_train, scaler)
validate_predict = inverse_normalize_data(validate_predict, scaler)
y_validate = inverse_normalize_data(y_validate, scaler)
x_test = data[84:]
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
x_test.shape
next_predict = model.predict(x_test)
next_predict_res = inverse_normalize_data(next_predict, scaler)
next_predict_res
next4_0 = np.pad(next_predict_res[:,:,1][0], [(4, 0)], mode='constant')
next4_0[:4]=y_validate[:,:,1][8]

next4_1 = np.pad(next_predict_res[:,:,2][0], [(4, 0)], mode='constant')
next4_1[:4]=y_validate[:,:,2][8]

next4_2 = np.pad(next_predict_res[:,:,3][0], [(4, 0)], mode='constant')
next4_2[:4]=y_validate[:,:,3][8]



BACK4_0 = np.pad(y_validate[:,:,1][8], [(0, 4)], mode='constant')
BACK4_0[4:]=np.NAN

BACK4_1 = np.pad(y_validate[:,:,2][8], [(0, 4)], mode='constant')
BACK4_1[4:]=np.NAN

BACK4_2 = np.pad(y_validate[:,:,3][8], [(0, 4)], mode='constant')
BACK4_2[4:]=np.NAN
day = ['04-2-2020','05-2-2020','06-2-2020','07-2-2020','08-2-2020','09-2-2020','10-2-2020','11-2-2020']

fig = plt.figure(figsize=(35, 20))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)                                            

ax1.plot(day,next4_0,color='blue',ls='--',clip_on=True,label='Predicted Confirmed')
ax1.plot(BACK4_0,color='blue',clip_on=True,label='Actual Confirmed')

ax1.title.set_text("Confirmed per/day")
ax1.legend()


ax2.bar(day,next4_1,color='maroon',ls='--',clip_on=True,label='Predicted Deaths')
ax2.bar(day,BACK4_1,color='blue',clip_on=True,label='Actual Deaths')
ax2.legend()

ax2.title.set_text("Deaths per/day")
ax2.legend()

ax3.bar(day,next4_2,color='maroon',ls='--',clip_on=True,label='Predicted Recoverd')
ax3.bar(day,BACK4_2,color='red',clip_on=True,label='Actual Recoverd')


ax3.title.set_text("Recoverd per/day")
ax3.legend()

plt.show()
dataf = pd.DataFrame(data=[day,next4_0, next4_1,next4_2])
dataf = dataf.T
dataf.columns=['Date','Confirmed','Deaths','Recoverd']

dataf.Confirmed = dataf.Confirmed.apply(lambda x : np.round(x,0))
dataf.Deaths = dataf.Deaths.apply(lambda x : np.round(x,0))
dataf.Recoverd = dataf.Recoverd.apply(lambda x : np.round(x,0))

dataf
