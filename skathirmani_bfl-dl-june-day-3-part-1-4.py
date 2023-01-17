# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales = pd.read_csv('/kaggle/input/trainings/sales_data.csv')
store = pd.read_csv('/kaggle/input/trainings/sales_store_data.csv')
sales_merged = pd.merge(left=sales, right=store, left_on='Store', right_on='Store', how='left')
sales_merged['CompetitionDistance'] = sales_merged['CompetitionDistance'].fillna(
    sales_merged['CompetitionDistance'].mean())
sales_merged['CompetitionOpenSinceYear'] = sales_merged['CompetitionOpenSinceYear'].fillna(
    sales_merged['CompetitionOpenSinceYear'].mean())
sales_merged['CompetitionOpenSinceMonth'] = sales_merged['CompetitionOpenSinceMonth'].fillna(
    sales_merged['CompetitionOpenSinceMonth'].mean())
sales_merged['Promo2SinceWeek'] = sales_merged['Promo2SinceWeek'].fillna(0)
sales_merged['Promo2SinceYear'] = sales_merged['Promo2SinceYear'].fillna(0)
sales_merged['PromoInterval'] = sales_merged['PromoInterval'].fillna(
    sales_merged['PromoInterval'].mode().iloc[0])

sales_merged['Date'] = pd.to_datetime(sales_merged['Date'], format='%Y-%m-%d')
sales_merged['month'] = sales_merged['Date'].dt.month
sales_merged['day'] = sales_merged['Date'].dt.day
sales_numerical = pd.get_dummies(sales_merged.drop(['Date', 'Customers'], axis=1), drop_first=True)
from sklearn.model_selection import train_test_split

input_cols = sales_numerical.drop(['Sales'],axis=1).columns
target_col = 'Sales'
train_x, test_x, train_y, test_y = train_test_split(sales_numerical[input_cols],
                                                    sales_numerical[target_col],
                                                    test_size=0.2,
                                                    random_state=1)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
# input_layers: 24 units
# hidden layer: 64 units, activation: relu
# o/p layer: 1 unit, activation: linear 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
train_x_scaled.shape
train_x_scaled.shape[0] *0.8 / 1000
model = Sequential()
model.add(layers.Dense(units=256, input_shape=(train_x.shape[1],), activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x_scaled, train_y, validation_split=0.2, epochs=50, batch_size=1024, verbose=0)
history.history.keys()
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss', 'Validation loss'])
from sklearn.metrics import mean_squared_error
yhat = model.predict(train_x_scaled)
mean_squared_error(train_y, yhat)
lr = [0, 0.001, 0.0001, 0.01]
optimizers = ['adam', 'rmsprop', 'sgd']

# grid search (0, 'adam'), (0.001, 'adam'),
# random search (0, 'sgd'), (0, 'rmsprop')
# bayesian search 
# 
### Epochs: early stopping criteria
import tensorflow as tf
early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model = Sequential()
model.add(layers.Dense(units=256, input_shape=(train_x.shape[1],), activation='relu'))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x_scaled, train_y, validation_split=0.2, epochs=50, batch_size=1024, verbose=0,
                   callbacks=[early_callback])
!pip install keras-tuner
from kerastuner import HyperModel
from kerastuner import RandomSearch
class SearchBestModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential()
        model.add(layers.Dense(units=hp.Int('units', 16, 32, 8), activation='relu', input_shape=input_shape)),
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

input_shape = (train_x.shape[1], )
hyper_model = SearchBestModel(input_shape)
tuner = RandomSearch(hyper_model, objective='mse', max_trials=3, executions_per_trial=2)
tuner.search(train_x_scaled, train_y, epochs=10, validation_split=0.2, verbose=0, batch_size=1024)
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
loss, mse = best_model.evaluate(test_x_scaled, test_y)
mse
np.sqrt(mse)
from tensorflow.keras import optimizers

class SearchBestModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential()
        model.add(layers.Dense(units=hp.Int('units', 16, 32, 8), activation='relu', input_shape=input_shape)),
        model.add(layers.Dense(1, activation='linear'))
        
        opt = optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4]))
        
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        return model

input_shape = (train_x.shape[1], )
hyper_model = SearchBestModel(input_shape)
tuner = RandomSearch(hyper_model, objective='mse', max_trials=3, executions_per_trial=2)
tuner.search(train_x_scaled, train_y, epochs=10, validation_split=0.2, verbose=1, batch_size=1024)
#from kerastuner import BayesianOptimization
model = Sequential()
model.add(layers.Dense(units=256, input_shape=(train_x.shape[1],), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x_scaled, train_y, validation_split=0.2, epochs=50, batch_size=1024, verbose=0,
                   callbacks=[early_callback])
yhat = model.predict(test_x_scaled)
np.sqrt(mean_squared_error(test_y, yhat))
