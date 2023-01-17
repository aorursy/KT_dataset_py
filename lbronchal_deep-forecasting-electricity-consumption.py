%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 
dataset = pd.read_csv("../input/dados_Brazil_GDP_Electricity.csv", index_col=1)
dataset = dataset.drop(dataset.columns[0], axis=1)
dataset.plot()

_=plt.show()
plt.scatter(x=dataset.iloc[:,0], y=dataset.iloc[:,1])

_=plt.show()
from keras.layers.core import Dense

from keras.layers.recurrent import LSTM

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

import math

import os
data = dataset.iloc[:,0].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

data = scaler.fit_transform(data)
NUM_TIMESTEPS = 5

max_elements = len(data) - NUM_TIMESTEPS - 1

X = np.zeros((data.shape[0], NUM_TIMESTEPS))

Y = np.zeros((data.shape[0], 1))

for i in range(len(data) - NUM_TIMESTEPS - 1):

    X[i] = data[i:i + NUM_TIMESTEPS].T

    Y[i] = data[i + NUM_TIMESTEPS + 1]

    

    

# reshape X to three dimensions (samples, timesteps, features)

X = np.expand_dims(X, axis=2)

#X = X.reshape(X.shape[0], X.shape[1], 1)



X = X[:max_elements]

Y = Y[:max_elements]
sp = int(0.7 * len(data))

Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
NUM_EPOCHS = 250

BATCH_SIZE = 5 



np.random.seed(123456)



# stateless

def build_model_stateless():

    model = Sequential()

    model.add(LSTM(10, input_shape=(NUM_TIMESTEPS, 1), return_sequences=False))

    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

    return model

    
model_stateless = build_model_stateless()



early_stopping = EarlyStopping(patience=2)

history = model_stateless.fit(Xtrain, Ytrain, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,

    validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping], verbose=0)
# summarize history for accuracy

plt.plot(history.history['mean_squared_error'])

plt.plot(history.history['val_mean_squared_error'])

plt.title('model mean squared error')

plt.ylabel('mean squared error')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

_=plt.show()
test_length = NUM_TIMESTEPS

dataset_train = dataset.iloc[0:-test_length,:]
last_point = dataset_train.iloc[-NUM_TIMESTEPS:, 0].values.reshape(1, NUM_TIMESTEPS)

last_point_scaled = scaler.transform(last_point)

last_point_scaled = last_point_scaled.reshape(1, NUM_TIMESTEPS, 1)
num_predictions = test_length + 10

year_ini = dataset_train.index[-1] + 1 
last_x = last_point_scaled

predictions = []

years = []

for i in range(num_predictions):

    pred_tmp = model_stateless.predict(last_x)

    pred = scaler.inverse_transform(pred_tmp)

    predictions.append(pred[0][0])

    years.append(year_ini+i)

    next_x = np.roll(last_x, 1)

    next_x[0,0] = pred_tmp

    last_x = next_x



predictions_df = pd.DataFrame({'Forecast':predictions}, index=years)    
ax = dataset.iloc[:,0].plot()

predictions_df.plot(ax=ax)

plt.ylabel("Electricity")

plt.show()
predictions_df.iloc[:test_length,:]
dataset.iloc[-test_length:, 0].values
pred_errors = dataset.iloc[-test_length:, 0].values - predictions_df.iloc[:test_length,0].values
np.sqrt(np.mean(pred_errors**2))