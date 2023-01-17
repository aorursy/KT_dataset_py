# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/nyse'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

from tqdm import tqdm



print(os.listdir("../input/nyse"))
df = pd.read_csv("../input/nyse/prices-split-adjusted.csv", header=0)

df.head(200)
symbols = list(set(df.symbol))

len(symbols)                        #extracting symbols
symbols[:20]
#taking a particular symbol for evaluating

google = df[df.symbol == 'GOOGL']

googles = df[df.symbol == 'GOOGL']

google = google.close.values.astype('float32')

google = google.reshape(1762, 1)

google.shape
import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots
plt.plot(google)

plt.show()



scaler = MinMaxScaler(feature_range=(0, 1))

google = scaler.fit_transform(google)
fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4])



fig.add_trace(go.Scatter(x=googles.date, y=googles.open.diff(), name='l1'),

              row=1, col=1)



fig.add_trace(go.Histogram(x=googles['open'].diff(), name='h1', histnorm='probability density'),

              row=1, col=2)

fig.update_layout( height=550, width=1130, title_text="Consecutive difference between opening stock price of Google shares")



fig.update_xaxes(title_text="Time", row=1, col=1);   fig.update_xaxes(title_text="Value", row=1, col=2)

fig.update_yaxes(title_text="Value", row=1, col=1);   fig.update_yaxes(title_text="Prob. Density", row=1, col=2)



fig.show()
f, axes= plt.subplots(2,2, figsize=(20,14))

sns.regplot(x=googles.open, y=googles.close, color="g", ax=axes[0][0])

sns.regplot(x=googles.open, y=googles.volume, ax=axes[0][1])

sns.regplot(x=googles.low, y=googles.high, color="b", ax=axes[1][0])

sns.regplot(x=googles.volume, y=googles.close, color="g", ax=axes[1][1])
f, axes= plt.subplots(1,2, figsize=(20,6))

sns.regplot(x=googles.open.diff(), y=googles.close.diff(), color="g", ax=axes[0])

sns.regplot(x=googles.low.diff(), y=googles.high.diff(), color="b", ax=axes[1])

plt.suptitle('Consecutive variation correlations', size=16)
corr= googles.corr()

plt.figure(figsize=(8,5))

sns.heatmap(corr, annot=True, cmap="Greens_r",linewidth = 3, linecolor = "white")
train_size = int(len(google) * 0.80)

test_size = len(google) - train_size

train, test = google[0:train_size,:], google[train_size:len(google),:]

print(len(train), len(test))
train = train.reshape(len(train) , 1)

test = test.reshape(len(test) , 1)
print(train.shape , test.shape)
def process_data(data , n_features):

    dataX, dataY = [], []

    for i in range(len(data)-n_features-1):

        a = data[i:(i+n_features), 0]

        dataX.append(a)

        dataY.append(data[i + n_features, 0])

    return np.array(dataX), np.array(dataY)
n_features = 2



trainX, trainY = process_data(train, n_features)

testX, testY = process_data(test, n_features)
print(trainX.shape , trainY.shape , testX.shape , testY.shape)
trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])

testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])
import numpy

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense , BatchNormalization , Dropout , Activation

from keras.layers import LSTM , GRU

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.optimizers import Adam , SGD , RMSprop
filepath="stock_weights.hdf5"

from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#building the model



model = Sequential()

model.add(GRU(256 , input_shape = (1 , n_features) , return_sequences=True))

model.add(Dropout(0.4))

model.add(LSTM(256))

model.add(Dropout(0.4))

model.add(Dense(64 ,  activation = 'relu'))

model.add(Dense(1))

print(model.summary())

# model = Sequential()



# model.add(LSTM(units=1, output_dim=50,return_sequences=True))

# model.add(Dropout(0.2))



# model.add(LSTM(100,return_sequences=False))

# model.add(Dropout(0.2))



# model.add(Dense(output_dim=1))

# model.add(Activation('linear'))



# start = time.time()

# model.compile(loss='mse', optimizer='rmsprop')

# print ('compilation time : ', time.time() - start)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005), metrics=['mean_squared_error'])
history = model.fit(trainX, trainY, epochs=100 , batch_size = 128 , 

          callbacks = [checkpoint , lr_reduce] , validation_data = (testX,testY))