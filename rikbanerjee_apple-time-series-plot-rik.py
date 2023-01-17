# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gujarat=pd.read_csv("/kaggle/input/statewise-apple-datarik/GApple.csv")

maharashtra=pd.read_csv("/kaggle/input/statewise-apple-datarik/MApple.csv")

harayana=pd.read_csv("/kaggle/input/statewise-apple-datarik/HApple.csv")

punjab=pd.read_csv("/kaggle/input/statewise-apple-datarik/PApple.csv")

mp=pd.read_csv("/kaggle/input/statewise-apple-datarik/MPApple.csv")

rajasthan=pd.read_csv("/kaggle/input/statewise-apple-datarik/RApple.csv")

tomato=pd.read_csv("../input/tomatopunjab/Tomato (1).csv")

state=[gujarat,maharashtra,harayana,punjab,mp,rajasthan,tomato]

for i in state:

    i.drop(["State","Commodity"],axis=1,inplace=True)

    i['Price Date'] = pd.to_datetime(i['Price Date'])

    i=i.set_index("Price Date")
"""

model=Sequential()

model.add(LSTM(50,return_sequences=True,input_shape=(50,1)))

model.add(LSTM(50,return_sequences=True))

model.add(LSTM(50))

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

"""
df1=tomato.reset_index()["Modal Price (Rs./Quintal)"]







import numpy as np

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

df1=scaler.fit_transform(np.array(df1).reshape(-1,1))



train_s=int(len(df1)*0.75)

test_s=len(df1)-train_s

train_data,test_data=df1[0:train_s,:],df1[train_s:len(df1),:1]





import numpy

def create_dataset(dataset,time_step=1):

    datax,datay=[],[]

    for i in range(len(dataset)-time_step-1):

        a = dataset[i:(i+time_step),0]

        datax.append(a)

        datay.append(dataset[i+time_step,0])

    return numpy.array(datax),numpy.array(datay)





time_step=60

x_train,y_train=create_dataset(train_data,time_step)

x_test,y_test=create_dataset(test_data,time_step)



x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM







model=Sequential()  

model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))

model.add(LSTM(50,return_sequences=True))

model.add(LSTM(50,return_sequences=True))

model.add(LSTM(50))

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=["mae","mape","acc"])
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2000,batch_size=64,verbose=1)
train_predict=model.predict(x_train)

test_predict=model.predict(x_test)
train_predict=scaler.inverse_transform(train_predict)

test_predict=scaler.inverse_transform(test_predict)

import math

from sklearn.metrics import mean_squared_error

math.sqrt(mean_squared_error(y_train,train_predict))
math.sqrt(mean_squared_error(y_test,test_predict))
plt.figure(figsize=(15,8))

look_back=time_step

trainPredictPlot=numpy.empty_like(df1)

trainPredictPlot[:,:] = np.nan

trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict

#shift test prediction for plotting

testPredictPlot=numpy.empty_like(df1)

testPredictPlot[:,:]=numpy.nan

testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict

#plot baseline and prediction

plt.plot(scaler.inverse_transform(df1))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.title("Blue=original,Yellow=how trains model,Green=Prediction  Punjab(epoch=2000 & time_step=60 & statcked lstm 4layer)")

plt.show()

predtomato=tomato[df1.shape[0]-test_predict.shape[0]:]

predtomato["predicted values"]=test_predict

predtomato.tail(30)
import plotly.offline as pyo

import plotly.graph_objs as go

pt=pd.pivot_table(predtomato,values=["Modal Price (Rs./Quintal)","predicted values"],index="Price Date")

trace0=go.Scatter(

    x=pt.index,

    y=pt["Modal Price (Rs./Quintal)"],

    mode="lines",

    name="TRUE VALUE"

)

trace1=go.Scatter(

    x=pt.index,

    y=pt["predicted values"],

    mode="lines",

    name="PREDICTED VALUE"

)

data=[trace0,trace1]

layout=go.Layout(title="TRUE VALUE/PREDICTION")

fig=go.Figure(data=data,layout=layout)

fig.update_xaxes(rangeslider_visible=True)

fig.show()
"""

from pandas.tseries.offsets import DateOffset

future_dates=[tomato.index[-1]+ DateOffset(months=x) for x in range(0,24)]

"""
len(test_data)

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

temp_input=list(x_input)
from numpy import array

lst_output=[]

n_steps=60

i=0

fu=300

while(i<fu):

    if(len(temp_input)>n_steps):

        x_input=np.array(temp_input[1:])

        #print("{} day input {}".format(i,x_input))

        x_input=x_input.reshape(1,-1)

        x_input=x_input.reshape(1,n_steps,1)

        yhat=model.predict(x_input,verbose=0)

        #print("{} day output {}".format(i,yhat))

        temp_input.extend(yhat[0].tolist())

        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())

        i+=1

    else:

        x_input=x_input.reshape((1,n_steps,1))

        yhat=model.predict(x_input,verbose=0)

        #print(yhat[0])

        temp_input.extend(yhat[0].tolist())

        #print(len(temp_input))

        lst_output.extend(yhat.tolist())

        i+=1

day_new=np.arange(1,time_step+1)

day_pred=np.arange(time_step+1,time_step+1+fu)

import matplotlib.pyplot as plt





plt.figure(figsize=(20,5))

df3=df1.tolist()

df3.extend(lst_output)

plt.plot(df3[100:])