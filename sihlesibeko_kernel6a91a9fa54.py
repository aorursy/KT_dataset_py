import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

from datetime import date

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("../input/login-time-for-users/data.csv",names=['count','UserID','LoginTime','IP'])[['UserID','LoginTime']]

data.head()

m= data.UserID == 413







logins = data[data.UserID == 413]

#logins[['Date','Time']] =logins.LoginTime.str.split(expand=True,)

logins = logins.drop("UserID", axis=1)

logins['LoginTime']=pd.to_datetime(logins['LoginTime'])

#logins=logins.loc[logins['LoginTime'].isin(['2017-01'])]

#logins = logins.groupby([logins['LoginTime'].dt.date])

#logins['Dates'] = pd.to_datetime(data['LoginTime'], format='%Y-%m-%d %H:%M:%S').dt.date

#logins['Hours'] = pd.to_datetime(data['LoginTime'], format='%Y-%m-%d %H:%M:%S').dt.time

#logins = logins.drop("LoginTime", axis=1)



logins.head()
#logins=logins[2017-1:2017-2]





date_from = pd.Timestamp(date(2017,1,1))

date_to = pd.Timestamp(date(2017,1,16))

logins = logins[(logins['LoginTime'] > date_from ) &(logins['LoginTime'] < date_to)]



logins.head(14)
vec1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec3=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec4=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec5=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec6=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec7=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec8=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec9=[0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0]

vec10=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]

vec11=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]

vec12=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec13=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec14=[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]

vec15=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

df=pd.DataFrame([vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,vec9,vec10,vec11,vec12,vec13,vec14,vec15])

df.head(16)

#defining test and train data

train,test=logins[:-12],logins[-12:]


_input=12 # for 12 days

_features= 24 # 24 hours

model= Sequential()

model.add(LSTM(200,activation="relu",input_shape=(_input,_features)))

loss = "binary_crossentropy"

#model.add(Dropout(0.4)) # to prevent overfitting

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(train,train, epochs=100, batch_size=64)

#model.fit_generator(generator,epochs=450)