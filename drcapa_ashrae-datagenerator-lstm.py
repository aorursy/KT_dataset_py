import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os

import random
from keras.utils import Sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding

from keras.optimizers import RMSprop,Adam

import keras.backend as K
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
path_in = '../input/ashrae-energy-prediction/'

print(os.listdir(path_in))
train_data = pd.read_csv(path_in+'train.csv', parse_dates=['timestamp'])

train_weather = pd.read_csv(path_in+'weather_train.csv', parse_dates=['timestamp'])

building_data = pd.read_csv(path_in+'building_metadata.csv')
def plot_bar(data, name):

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    data_label = data[name].value_counts()

    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))

    names = list(dict_train.keys())

    values = list(dict_train.values())

    plt.bar(names, values)

    ax.set_xticklabels(names, rotation=45)

    plt.grid()

    plt.show()
cols_with_missing_train_weather = [col for col in train_weather.columns if train_weather[col].isnull().any()]

cols_with_missing_building = [col for col in building_data.columns if building_data[col].isnull().any()]
print(cols_with_missing_train_weather)

print(cols_with_missing_building)
imp_most = SimpleImputer(strategy='most_frequent')

train_weather[cols_with_missing_train_weather] = imp_most.fit_transform(train_weather[cols_with_missing_train_weather])

building_data[cols_with_missing_building] = imp_most.fit_transform(building_data[cols_with_missing_building])
train_data['meter_reading'] = np.log1p(train_data['meter_reading'])
train_data['month'] = train_data['timestamp'].dt.month

train_data['day'] = train_data['timestamp'].dt.weekday

train_data['year'] = train_data['timestamp'].dt.year

train_data['hour'] = train_data['timestamp'].dt.hour
train_data['weekend'] = np.where((train_data['day'] == 5) | (train_data['day'] == 6), 1, 0)
train_weather['wind_direction'+'_sin'] = np.sin((2*np.pi*train_weather['wind_direction'])/360)

train_weather['wind_direction'+'_cos'] = np.cos((2*np.pi*train_weather['wind_direction'])/360)

train_weather = train_weather.drop(['wind_direction'], axis=1)
train_data = pd.get_dummies(train_data, columns=['meter'])
features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}

for feature in features_cyc.keys():

    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])

    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])

train_data = train_data.drop(features_cyc.keys(), axis=1)
plot_bar(building_data, 'primary_use')
map_use = dict(zip(building_data['primary_use'].value_counts().sort_index().keys(),

                     range(1, len(building_data['primary_use'].value_counts())+1)))
building_data['primary_use'] = building_data['primary_use'].replace(map_use)
#building_data = pd.get_dummies(building_data, columns=['primary_use'])
weather_scale = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'sea_level_pressure', 'wind_speed']
mean = train_weather[weather_scale].mean(axis=0)

train_weather[weather_scale] = train_weather[weather_scale].astype('float32')

train_weather[weather_scale] -= train_weather[weather_scale].mean(axis=0)

std = train_weather[weather_scale].std(axis=0)

train_weather[weather_scale] /= train_weather[weather_scale].std(axis=0)
building_scale = ['square_feet', 'year_built', 'floor_count']
mean = building_data[building_scale].mean(axis=0)

building_data[building_scale] = building_data[building_scale].astype('float32')

building_data[building_scale] -= building_data[building_scale].mean(axis=0)

std = building_data[building_scale].std(axis=0)

building_data[building_scale] /= building_data[building_scale].std(axis=0)
train_data = pd.merge(train_data, building_data, on='building_id', right_index=True)

train_data = train_data.sort_values(['timestamp'])

train_data = pd.merge_asof(train_data, train_weather, on='timestamp', by='site_id', right_index=True)

del train_weather
class DataGenerator(Sequence):

    """ A data generator based on the template

        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        """

    

    def __init__(self, data, list_IDs, features, batch_size, shuffle=False):

        self.data = data.loc[list_IDs].copy()

        self.list_IDs = list_IDs

        self.features = features

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.on_epoch_end()

    

    

    def __len__(self):

        return int(np.floor(len(self.list_IDs)/self.batch_size))

    

    

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



        

    def __data_generation(self, list_IDs_temp):        

        X = np.empty((len(list_IDs_temp), len(self.features)), dtype=float)

        y = np.empty((len(list_IDs_temp), 1), dtype=float)

        X = self.data.loc[list_IDs_temp, self.features].values

        

        if 'meter_reading' in self.data.columns:

            y = self.data.loc[list_IDs_temp, 'meter_reading'].values

        # reshape

        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        return X, y
train_size = int(len(train_data.index)*0.75)

val_size = len(train_data.index) - train_size

train_list, val_list = train_data.index[0:train_size], train_data.index[train_size:train_size+val_size]

print(train_size, val_size)

no_features = ['building_id', 'timestamp', 'meter_reading', 'year']

features = train_data.columns.difference(no_features)
batch_size = 1024

train_generator = DataGenerator(train_data, train_list, features, batch_size)

val_generator = DataGenerator(train_data, val_list, features, batch_size)
input_dim = len(features)

print(input_dim)
model = Sequential()

#model.add(Embedding(input_length=input_dim))

model.add(LSTM(units=8, activation = 'relu', input_shape=(1, input_dim)))

#model.add(LSTM(units=64, activation = 'relu'))

#model.add(Dense(128, activation='relu', input_dim=input_dim))

#model.add(Dense(256, activation='relu'))

#model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='relu'))
def rmse(y_true, y_pred):

    """ root_mean_squared_error """

    return K.sqrt(K.mean(K.square(y_pred - y_true)))
model.compile(optimizer = Adam(lr=1e-4),

              loss='mse',

              metrics=[rmse])
model.summary()
epochs = 1
history = model.fit_generator(generator=train_generator,

                              validation_data=val_generator,

                              epochs = epochs)
loss = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='loss_train')

plt.plot(epochs, loss_val, 'b', label='loss_val')

plt.title('value of the loss function')

plt.xlabel('epochs')

plt.ylabel('value of the loss function')

plt.legend()

plt.grid()

plt.show()
acc = history.history['rmse']

acc_val = history.history['val_rmse']

epochs = range(1, len(loss)+1)

plt.plot(epochs, acc, 'bo', label='accuracy_train')

plt.plot(epochs, acc_val, 'b', label='accuracy_val')

plt.title('accuracy')

plt.xlabel('epochs')

plt.ylabel('value of accuracy')

plt.legend()

plt.grid()

plt.show()
del train_data
nrows = 1667904

batch_size = 1022

steps = 25

y_test = np.empty(())

test_weather = pd.read_csv(path_in+'weather_test.csv', parse_dates=['timestamp'])

cols_with_missing_test_weather = [col for col in test_weather.columns if test_weather[col].isnull().any()]

test_weather[cols_with_missing_test_weather] = imp_most.fit_transform(test_weather[cols_with_missing_test_weather])



mean = test_weather[weather_scale].mean(axis=0)

test_weather[weather_scale] = test_weather[weather_scale].astype('float32')

test_weather[weather_scale] -= test_weather[weather_scale].mean(axis=0)

std = test_weather[weather_scale].std(axis=0)

test_weather[weather_scale] /= test_weather[weather_scale].std(axis=0)



test_weather['wind_direction'+'_sin'] = np.sin((2*np.pi*test_weather['wind_direction'])/360)

test_weather['wind_direction'+'_cos'] = np.cos((2*np.pi*test_weather['wind_direction'])/360)

test_weather = test_weather.drop(['wind_direction'], axis=1)



for i in range(0, steps):

    print('work on step ', (i+1))

    test_data = pd.read_csv(path_in+'test.csv', skiprows=range(1,i*(nrows)+1), nrows=nrows, parse_dates=['timestamp'])

    test_data['month'] = test_data['timestamp'].dt.month

    test_data['day'] = test_data['timestamp'].dt.weekday

    test_data['year'] = test_data['timestamp'].dt.year

    test_data['hour'] = test_data['timestamp'].dt.hour

    test_data['weekend'] = np.where((test_data['day'] == 5) | (test_data['day'] == 6), 1, 0)

    for feature in features_cyc.keys():

        test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/features_cyc[feature])

        test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/features_cyc[feature])

    test_data = test_data.drop(features_cyc.keys(), axis=1)

    test_data = pd.get_dummies(test_data, columns=['meter'])

    test_data = pd.merge(test_data, building_data, on='building_id', right_index=True)

    test_data = test_data.sort_values(['timestamp'])

    test_data = pd.merge_asof(test_data, test_weather, on='timestamp', by='site_id', right_index=True)

    test_data = test_data.sort_values(['row_id'])

    for feature in features:

        if feature not in test_data:

            #print('   not in:', feature)

            test_data[feature] = 0

    test_generator = DataGenerator(test_data, test_data.index, features, batch_size)

    predict = model.predict_generator(test_generator, verbose=1, workers=1)

    predict = np.expm1(predict)

    y_test = np.vstack((y_test, predict))

    del test_data

    del test_generator
y_test = np.delete(y_test, 0, 0)
del test_weather

del building_data
output = pd.DataFrame({'row_id': range(0, len(y_test)),

                       'meter_reading': y_test.reshape(len(y_test))})

output = output[['row_id', 'meter_reading']]

output.to_csv('submission.csv', index=False)