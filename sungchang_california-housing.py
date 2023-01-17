import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
df
df.describe()
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])
df
data = df
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.3, vmin=-0.3,linewidths=1)
plt.show()
X_ = df.dropna(axis=0)
Y_ = X_[['total_bedrooms']]
X_ = X_.drop('total_bedrooms', axis=1)
train_x = X_.iloc[:].values
train_y = Y_.iloc[:].values
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc1 = MinMaxScaler()
sc2 = MinMaxScaler()
normalX = sc1.fit_transform(train_x)
normalY = sc2.fit_transform(train_y)
train_x = normalX
train_y = normalY
#from sklearn.model_selection import train_test_split
#train_x, test_x, train_y, test_y = train_test_split(normalX,normalY,test_size=0.99, random_state = 42)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, LSTM, ConvLSTM2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers

NNinput = train_x.shape[1]
act = 'relu'
opt = 'Adam'
los = 'mean_squared_error'

model = Sequential()
model.add(Dense(128, activation = act, input_shape = [NNinput,]))
model.add(Dense(128, activation = act))
model.add(Dense(128, activation = act))
model.add(Dense(1, activation = act))
model.compile(optimizer = opt, loss = los, metrics = ['mse'])
#model.summary()
batch_size = 50
epoch = 5
history = model.fit(train_x, train_y, epochs = epoch, batch_size = batch_size, verbose = 1)
X_ = df
Y_ = X_[['total_bedrooms']]
X_ = X_.drop('total_bedrooms', axis=1)

test_x = X_.iloc[:].values
test_y = Y_.iloc[:].values
normalX = sc1.transform(test_x)
normalY = sc2.transform(test_y)
test_x = normalX
test_y = normalY
pred = model.predict(test_x)
answer = np.concatenate((pred, test_y), axis=1)
answer = pd.DataFrame(answer)
answer.columns = ['pred', 'test']
answer[285:295]
answer['total_bedrooms'] = np.where(answer['test'].isnull(),answer['pred'],answer['test'])
answer['total_bedrooms'] = sc2.inverse_transform(answer[['total_bedrooms']])
answer
df_col = df.columns
df = df.drop('total_bedrooms', axis=1)
df['total_bedrooms'] = answer['total_bedrooms']
df = df[df_col]
df[285:295]
X_ = df
Y_ = X_[['median_income']]
X_ = X_.drop('median_income', axis=1)

test_x = X_.iloc[:].values
test_y = Y_.iloc[:].values
normalX = sc1.fit_transform(test_x)
normalY = sc2.fit_transform(test_y)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(normalX,normalY,test_size=0.3, random_state = 42)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, LSTM, ConvLSTM2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers

NNinput = train_x.shape[1]
act = 'relu'
opt = 'Adam'
los = 'mean_squared_error'

model = Sequential()
model.add(Dense(128, activation = act, input_shape = [NNinput,]))
model.add(Dense(128, activation = act))
model.add(Dense(128, activation = act))
model.add(Dense(1, activation = act))
model.compile(optimizer = opt, loss = los, metrics = ['mse'])
#model.summary()
batch_size = 128
epoch = 50
history = model.fit(train_x, train_y, epochs = epoch, batch_size = batch_size, verbose = 1)
pred = model.predict(test_x)
pred1 = sc2.inverse_transform(pred)
test_y1 = sc2.inverse_transform(test_y)

1 - abs(1 - test_y1 / pred1).mean()
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred1[: ,])
plt.plot(test_y1[: ,])
plt.show()
