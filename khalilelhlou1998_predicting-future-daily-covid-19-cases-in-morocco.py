!pip install COVID19Py
import pandas as pd

import numpy as np

import plotly.express as px  

import plotly.graph_objects as go

#################################

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime

import COVID19Py
covid19 = COVID19Py.COVID19()

timeline = covid19.getLocationById(163)['timelines']['confirmed']['timeline']



dic = { 'Date' : list(timeline.keys()),

        'Cases': list(timeline.values())}



df = pd.DataFrame.from_dict(dic)

df
fig = go.Figure(data=go.Scatter(x=df.Date, y=df.Cases, mode='lines+markers'))

fig.show()
corona = df.copy()

# Drop the first 37 rows

corona = corona[37:]

corona = corona.reset_index(drop=True)

#undo the accumulation.

corona = corona.set_index('Date')

corona = corona.diff().fillna(0).astype(np.int64)

corona
fig = go.Figure(data=go.Scatter(x=corona.index, y=corona.Cases, mode='lines+markers'))

fig.show()
corona.shape
test_data_size = 20



train_data = corona[:-test_data_size]

test_data = corona[-test_data_size:]
train_data.shape
test_data.shape
train_set = train_data.values

test_set = test_data.values
test_set[:2]
#Initialising the MinMaxscaler ()

scaler = MinMaxScaler(feature_range = (0, 1))



#Transforming training and test values 

train_set = scaler.fit_transform(train_set)

test_set = scaler.fit_transform(test_set)
test_set[:2]
def sequences(data, seq_length):

    X_values = []

    Y_label = []

    for i in range(seq_length, len(data)):

        X_values.append(data[i-seq_length:i, 0])

        Y_label.append(data[i, 0])

    return np.array(X_values), np.array(Y_label)
#Creat sequences

train_X, train_Y = sequences(train_set, seq_length=7)

test_X, test_Y= sequences(test_set, seq_length=7)
test_X[:2]
#Reshapin 

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
test_X[:2]
test_Y[:2]
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))

model.add(Dropout(0.3))



model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.3))



model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.3))



model.add(LSTM(units=50))

model.add(Dropout(0.3))



model.add(Dense(units = 1))



model.compile(optimizer = 'adam', loss = 'mean_squared_error')



history = model.fit(train_X, train_Y, epochs = 100, validation_data=(test_X,test_Y))
fig = go.Figure()

fig.add_trace(go.Scatter(x=[i for i in range(1,54)], y=history.history['loss'],mode='lines+markers',name='Training loss'))

fig.add_trace(go.Scatter(x=[i for i in range(1,54)], y=history.history['val_loss'],mode='lines+markers',name='Valid loss'))

fig.show()
test_inputs = corona[-test_data_size-7:].values

test_inputs = test_inputs.reshape(-1,1)

test_inputs = scaler.transform(test_inputs)
features_X , features_Y =  sequences(test_inputs, seq_length=7)
features_X = np.array(features_X)

features_X = np.reshape(features_X, (features_X.shape[0], features_X.shape[1], 1))



features_Y = np.array(features_Y)

features_Y = np.reshape(features_Y, (features_X.shape[0], 1))
predictions = model.predict(features_X)
features_Y = scaler.inverse_transform(features_Y)

predictions = scaler.inverse_transform(predictions)
features = [list(i)[0] for i in list(features_Y )]

predict = [list(i)[0] for i in list(predictions)]
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona.index[:len(train_data)], y= corona.Cases[:len(train_data)], mode='lines+markers',name='Historical Daily Cases'))

fig.add_trace(go.Scatter(x=corona.index[-len(test_data):],y=features , mode='lines+markers',name='Real Daily Cases'))

fig.add_trace(go.Scatter(x=corona.index[-len(test_data):], y=predict, mode='lines+markers',name='Predicted Daily Cases'))

fig.show()
DAYS_TO_PREDICT = 7
predict_values = []

data = corona.Cases[-DAYS_TO_PREDICT:].values

for _ in range (DAYS_TO_PREDICT):

    

    X = data.reshape(-1,1)

    X = scaler.fit_transform(X)

    X_values =[]

    X_values.append(X[:7])

    # 7 is the sequence lenght 

    X = np.reshape(X_values,(1,7,1))



    day_prediction = model.predict(X)

    #As before, we'll inverse the scaler transformation

    day_prediction = scaler.inverse_transform(day_prediction)

    predict_values.append(day_prediction[0][0])

    

    data = np.append(data, day_prediction)

    data = np.delete(data,0)
next_days = pd.date_range(

  start=corona.index[-1],

  periods=DAYS_TO_PREDICT + 1,

  closed='right'

)

next_days.strftime("%d/%m/%Y")

fig = go.Figure(data=[go.Table(header=dict(values=[i for i in next_days.strftime("%d/%m/%Y")]),

                 cells=dict(values=[int(i) for i in predict_values]))

                     ])

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona.index, y= corona.Cases, mode='lines+markers',name='Historical Daily Cases'))

fig.add_trace(go.Scatter(x=next_days,y=predict_values , mode='lines+markers',name='Future Cases'))



fig.show()