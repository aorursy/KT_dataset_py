import pandas as pd
import numpy as np
import random as rnd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
from matplotlib import pyplot
from datetime import datetime

traffic_data = pd.read_csv("traffic_volume_1hr.csv")
traffic_data
array_of_dtobjects = []
traffic_data["hr-1"] = traffic_data["hr"] - 1
for i in range(len(traffic_data)):
    array_of_dtobjects.append(datetime(year=2020,month=traffic_data["month"][i],day=traffic_data["day"][i],hour=traffic_data["hr-1"][i]))
    
traffic_data["dt_object"] = array_of_dtobjects

traffic_data
pyplot.figure(figsize=(12, 10))
plot_data = traffic_data[(traffic_data.month == 1) & (traffic_data.day <= 7)]
fig, ax = pyplot.subplots(figsize=(10,7))
plot_data.set_index('hr', inplace=True)
pt = plot_data.groupby('day')['total_volume'].plot(legend=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
traffic_data["total_volume"] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(traffic_data["total_volume"])))
traffic_data_month_five = traffic_data[traffic_data["month"]==5]
traffic_data_training = traffic_data[traffic_data["month"]!=5]

traffic_data_training
steps = 24
def prepareInputXAndInputY(input_data,steps):
    all_data_x = []
    all_data_y = []
    for i in range(0, len(input_data)-steps):
        data_x = input_data.iloc[i:i+steps,3]
        data_y = input_data.iloc[i+steps,3]
        data_x_converted = np.asarray([[xi] for xi in data_x.to_numpy() ])
        all_data_x.append(data_x_converted)
        all_data_y.append(data_y)
    
    all_data_x = np.asarray(all_data_x)
    all_data_y = np.asarray(all_data_y)
    return (all_data_x,all_data_y)
sgd=SGD(lr=0.1)
model = Sequential()
model.add(LSTM(7, input_shape=( steps, 1 ), stateful=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=sgd)
model.summary()
from keras.utils import plot_model
plot_model(model, show_shapes=True)
all_data_x,all_data_y = prepareInputXAndInputY(traffic_data_training,steps)
model.fit(all_data_x, all_data_y, epochs=10, batch_size=5, verbose=2, shuffle=True)

test_data_x,test_data_y = prepareInputXAndInputY(traffic_data_month_five,steps)

trainPredict = model.predict(all_data_x)
testPredict = model.predict(test_data_x)


trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

pyplot.figure(figsize=(15, 15))
original_data_training = scaler.inverse_transform(traffic_data_training[["total_volume"]])
pyplot.plot(traffic_data_training[["dt_object"]][steps:steps+100],trainPredict[:100])
pyplot.plot(traffic_data_training[["dt_object"]][:100],original_data_training[:100],color="Yellow")

pyplot.show()
pyplot.figure(figsize=(15, 15))
original_data = scaler.inverse_transform(traffic_data_month_five[["total_volume"]])
pyplot.plot(traffic_data_month_five[["dt_object"]][steps:steps+100],testPredict[:100])
pyplot.plot(traffic_data_month_five[["dt_object"]][:100],original_data[:100],color="Yellow")

pyplot.show()