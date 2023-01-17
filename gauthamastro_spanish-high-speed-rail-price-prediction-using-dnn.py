import pandas as pd

import numpy as np

import pickle

import datetime

import math

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from keras.layers import Input,Dense

from keras.models import Model

from keras.optimizers import Nadam

from keras.callbacks import ModelCheckpoint
df = pd.read_csv('../input/renfe.csv')

df.head()
df.isna().sum() #Number of null values in different columns
df.dropna(inplace=True) #Remove all the rows containing nan
df.drop(["Unnamed: 0"], axis = 1, inplace = True)

df.drop(["insert_date"], axis = 1, inplace = True) #These values doesn't effect

df.head()
df["origin"].value_counts().plot(kind = "bar") #Origin places distribution
df["destination"].value_counts().plot(kind = "bar")
df['destination'].replace(["MADRID","BARCELONA",'SEVILLA','VALENCIA','PONFERRADA'],[0,1,2,3,4], inplace = True) #Maps to numerical values

df['origin'].replace(["MADRID","BARCELONA",'SEVILLA','VALENCIA','PONFERRADA'],[0,1,2,3,4], inplace = True)
df["train_type"].value_counts().plot(kind = "bar")

k = df["train_type"].unique()

l = [x for x in range(len(k))]

print("Numbers used to encode different train types",l) #numbers used to encode different trains

df['train_type'].replace(k,l, inplace = True)
df["train_class"].value_counts().plot(kind = "bar")#Plotting for train classes

k = df["train_class"].unique()

l = [x for x in range(len(k))]

print("Numbers used to encode different train classes:",l)

df['train_class'].replace(k,l, inplace = True)
df["fare"].value_counts().plot(kind = "bar") #Plotting for fare types

k = df["fare"].unique()

l = [x for x in range(len(k))]

print("Numbers used to encode different fare classes:",l)

df['fare'].replace(k,l, inplace = True)
f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(df.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)

plt.show()
start = df['start_date'].values

end = df['end_date'].values

datetimeFormat = '%Y-%m-%d %H:%M:%S'

duration = []

for i in range(len(start)):

    diff = datetime.datetime.strptime(end[i], datetimeFormat)- datetime.datetime.strptime(start[i], datetimeFormat)

    duration.append(diff.seconds)

df['duration'] = duration
start_weekdays = []

for i in start:

    start_weekdays.append(datetime.datetime.strptime(i,datetimeFormat).weekday())

end_weekdays = []

for i in end:

    end_weekdays.append(datetime.datetime.strptime(i,datetimeFormat).weekday())

df['start_weekday'] = start_weekdays

df['end_weekday'] = end_weekdays
#Converting datetime to cyclic features for departure times

hr_cos = [] #hr_cos,hr_sin,min_cos,min_sin

hr_sin = []

min_cos = []

min_sin = []

data = df['start_date'].values

for i in range(len(data)):

    time_obj = datetime.datetime.strptime(data[i],'%Y-%m-%d %H:%M:%S')

    hr = time_obj.hour

    minute = time_obj.minute

    sample_hr_sin = math.sin(hr*(2.*math.pi/24))

    sample_hr_cos = math.cos(hr*(2.*math.pi/24))

    sample_min_sin = math.sin(minute*(2.*math.pi/60))

    sample_min_cos = math.cos(minute*(2.*math.pi/60))

    hr_cos.append(sample_hr_cos)

    hr_sin.append(sample_hr_sin)

    min_cos.append(sample_min_cos)

    min_sin.append(sample_min_sin)

df['depart_time_hr_sin'] = hr_sin

df['depart_time_hr_cos'] = hr_cos

df['depart_time_min_sin'] = min_sin

df['depart_time_min_cos'] = min_cos

#Converting datetime to cyclic features for arrival times

hr_cos = [] #hr_cos,hr_sin,min_cos,min_sin

hr_sin = []

min_cos = []

min_sin = []

data = df['end_date'].values

for i in range(len(data)):

    time_obj = datetime.datetime.strptime(data[i],'%Y-%m-%d %H:%M:%S')

    hr = time_obj.hour

    minute = time_obj.minute

    sample_hr_sin = math.sin(hr*(2.*math.pi/24))

    sample_hr_cos = math.cos(hr*(2.*math.pi/24))

    sample_min_sin = math.sin(minute*(2.*math.pi/60))

    sample_min_cos = math.cos(minute*(2.*math.pi/60))

    hr_cos.append(sample_hr_cos)

    hr_sin.append(sample_hr_sin)

    min_cos.append(sample_min_cos)

    min_sin.append(sample_min_sin)

df['arrival_time_hr_sin'] = hr_sin

df['arrival_time_hr_cos'] = hr_cos

df['arrival_time_min_sin'] = min_sin

df['arrival_time_min_cos'] = min_cos

df.drop(["start_date"], axis = 1, inplace = True)

df.drop(["end_date"], axis = 1, inplace = True)
f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(df.corr(), annot=True, cmap = "Greens", linewidths=.5, fmt= '.2f',ax = ax)

plt.show()
df.head()#Just checking to see if all worked as per our logic
places_sc = MinMaxScaler(copy=False)

train_type_sc = MinMaxScaler(copy=False)

train_class_sc = MinMaxScaler(copy=False)

fare_sc = MinMaxScaler(copy=False)

weekday_sc = MinMaxScaler(copy=False)

duration_sc = MinMaxScaler(copy=False)

price_sc = MinMaxScaler(copy=False)

df['origin'] = places_sc.fit_transform(df['origin'].values.reshape(-1,1))

df['destination'] = places_sc.fit_transform(df['destination'].values.reshape(-1,1))

df['train_type'] = train_type_sc.fit_transform(df['train_type'].values.reshape(-1,1))

df['train_class'] = train_class_sc.fit_transform(df['train_class'].values.reshape(-1,1))

df['fare'] = fare_sc.fit_transform(df['fare'].values.reshape(-1,1))

df['start_weekday'] = weekday_sc.fit_transform(df['start_weekday'].values.reshape(-1,1))

df['end_weekday'] = weekday_sc.fit_transform(df['end_weekday'].values.reshape(-1,1))

df['duration'] = duration_sc.fit_transform(df['duration'].values.reshape(-1,1))

df['price'] = price_sc.fit_transform(df['price'].values.reshape(-1,1))
df.head()
data = df.values

Y = data[:,3]

X = np.delete(data,3,1)

#Creating different data splits for training,validation and testing!

x_train = X[:2223708]

y_train = Y[:2223708]

x_validation = X[2223708:2246398]

y_validation = Y[2223708:2246398]

x_test = X[2246398:]

y_test = Y[2246398:]
input_layer = Input((X.shape[1],))

y = Dense(64,kernel_initializer='he_normal',activation='tanh')(input_layer)

y = Dense(8,kernel_initializer='he_normal',activation='sigmoid')(y)

y = Dense(1,kernel_initializer='he_normal',activation='sigmoid')(y)

y = Dense(1,kernel_initializer='he_normal',activation='tanh')(y)

model = Model(inputs=input_layer,outputs=y)

model.compile(Nadam(),loss='mse')

model.summary()
history = model.fit(x_train,y_train,validation_data=(x_validation,y_validation),epochs = 100,batch_size=2048,callbacks=[ModelCheckpoint('best_model.hdf5',monitor='val_loss',mode='min')])
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend()

plt.show()
model.load_weights('best_model.hdf5') #this will load the best model that we saved earlier

scores = model.evaluate(x_test,y_test)

print("Test Set  RMSE(before scaling ):",scores)

pred = model.predict(x_test)

y_test = y_test.reshape(22692,1)

k = y_test-pred

k = price_sc.inverse_transform(k)

rmse = np.sqrt(np.mean(np.square((k))))

print('Test Set RMSE(after scaling) :',rmse)