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
import pandas as pd

df1 = pd.read_csv ("/kaggle/input/june15.csv")

df2 = pd.read_csv ("/kaggle/input/july15.csv")

df3 = pd.read_csv ("/kaggle/input/aug15.csv")

df4 = pd.read_csv ("/kaggle/input/sep15.csv")

df5 = pd.read_csv ("/kaggle/input/oct15.csv")

# df1=pd.read_csv("june15.csv")

# /kaggle/input/june15.csv

# df = pd.read_csv (r'Path where the CSV file is stored\File name.csv')

# print (df)

df = pd.concat([df1,df2,df3,df4,df5],axis=0)
df.head()
df1.head()


# df1.drop(axis=1, columns="total")



# print (df1)
# # import plotly.graph_objects as go

# import plotly.express as px





# fig = px.line(df1, x='timestamp', y='mb')

# fig.show()
df1['timestamp'] = pd.to_datetime(df1['timestamp'])  # Makes sure your timestamp is in datetime format

df1=df1.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df1.head()
df1.reset_index(level=0, inplace=True)
df1
import plotly.express as px



fig = px.line(df1, x='timestamp', y='mb')

fig.show()
df2['timestamp'] = pd.to_datetime(df2['timestamp'])  # Makes sure your timestamp is in datetime format

df2=df2.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df2.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df2, x='timestamp', y='mb')

fig.show()
df3['timestamp'] = pd.to_datetime(df3['timestamp'])  # Makes sure your timestamp is in datetime format

df3=df3.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df3.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df3, x='timestamp', y='mb')

fig.show()
df4['timestamp'] = pd.to_datetime(df4['timestamp'])  # Makes sure your timestamp is in datetime format

df4=df4.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df4.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df4, x='timestamp', y='mb')

fig.show()
df5['timestamp'] = pd.to_datetime(df5['timestamp'])  # Makes sure your timestamp is in datetime format

df5=df5.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df5.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df5, x='timestamp', y='mb')

fig.show()
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Makes sure your timestamp is in datetime format

df=df.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()

df.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df, x='timestamp', y='mb')

fig.show()
#imports

import tensorflow as tf

print(tf.__version__)

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PowerTransformer,MinMaxScaler
#playground

from keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np

data = np.array([i for i in range(20)])

data = np.append(data,[0])

targets = np.array([[data[i],data[i+1]] for i in range(1,data.shape[0])])

print(targets)

# for i in range(data.shape[0]-targets.shape[0]):

#      targets = np.insert(targets,0,[0,0],axis=0)

print(data.shape)

print(targets.shape)

data_gen = TimeseriesGenerator(data, targets,

                               length=3,batch_size=1)

for i in range(len(data_gen)):

    x, y = data_gen[i]

    print('%s => %s' % (x, y))
#playground

# define dataset

series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# define generator

n_input = 2

generator = TimeseriesGenerator(series, series, length=n_input, batch_size=1)

# number of samples

print('Samples: %d' % len(generator))

# print each sample

for i in range(len(generator)):

    x, y = generator[i]

    print('%s => %s' % (x, y))
def get_generator(data,previous_steps=24,forward_steps=1,batch_size=1,verbose=False):

    """

    @params

    data: numpy array of shape (n_records,1)

    """

    print(data.shape)

    data_copy = data.copy()

    data_copy = np.append(data_copy,np.zeros((1,1)),axis=0)

    targets = np.array([data_copy[i:i+forward_steps] for i in range(0,data.shape[0])])

    #print(data)

    #print(targets)

    generator = TimeseriesGenerator(data,targets,length=previous_steps,batch_size=batch_size)

    for i in range(5):

        x, y = generator[i]

        if verbose:

            print('%s => %s' % (x, y))

    return generator
#testing get generator

test_data = np.arange(10)[:,np.newaxis]

get_generator(test_data,3,2,1,True)

# test_data = df1.head(10).mb.values[:,np.newaxis]

# get_generator(test_data,3,2,1,True)
data = df.copy()

normalizer = MinMaxScaler()

#normalize data

print("Normalizing data using ",normalizer)

print(data.mb.head())

data.mb = normalizer.fit_transform(data.mb.values[:,np.newaxis])

print(data.mb.head())

ratio = 0.85

train_index = int(data.mb.values.shape[0] * ratio)

input_data = data.mb.values[:,np.newaxis]

train_data = input_data[:train_index]

test_data = input_data[train_index:]

#preprocessing into X's and y's

train_series_generator = get_generator(train_data)

test_series_generator = get_generator(test_data)

train_ds = tf.data.Dataset.from_generator(train_series_generator.__iter__,(tf.float32,tf.float32),((1,None,1),(1,None,1)))

test_ds = tf.data.Dataset.from_generator(test_series_generator.__iter__,(tf.float32,tf.float32),((1,None,1),(1,None,1)))

#define model

input_layer = keras.layers.Input(shape=(None,1))

lstm_layer_1 = keras.layers.LSTM(64,activation='relu',return_sequences=True)(input_layer)

lstm_layer = keras.layers.LSTM(128,activation='relu')(lstm_layer_1)

out_layer = keras.layers.Dense(1)(lstm_layer)

model = keras.Model(inputs=input_layer,outputs=out_layer)

print(model.summary())

#build model

model.compile(loss='mse',optimizer='adam',metrics=['mse'])

#fit model

# model.fit_generator(train_series_generator,epochs=64)

model.fit(train_ds,epochs=10,validation_data=test_ds)

#report results

#return model