import pandas as pd
from sklearn.datasets import load_boston
boston_data = load_boston()
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston['MEDV'] = boston_data.target
boston.head()
X = boston.drop(columns=['RAD','MEDV'])
Y = boston['MEDV']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
layers.Dense(64, activation='relu'),
layers.Dense(1)
])
model.summary()
model.compile(loss='mean_squared_error',
                optimizer='adam', metrics = "rmse")
early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)
model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(X_train,Y_train,
                    batch_size=32,
                    epochs=250,
                    callbacks= [early_stop],
                    validation_data=(X_test,Y_test))
import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.yscale('log')
plt.show()
final = model.predict(X_test)
from sklearn.metrics import mean_squared_error
import numpy as np
final = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, final)))
print('RMSE is {}'.format(rmse))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = keras.Sequential([
layers.Dense(84, activation='relu', input_shape=[len(X.keys())]),
layers.Dense(84, activation='relu'),
layers.Dense(1)
])
model.summary()

from keras import backend
 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
model.compile(optimizer='adam',
              loss='mean_squared_error', metrics=[rmse])
history = model.fit(X_train,Y_train,
                    batch_size=32,
                    epochs=100,
                    verbose=0,
                    callbacks= [early_stop],
                    validation_data=(X_test,Y_test))
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
from sklearn.metrics import mean_squared_error
import numpy as np
final = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, final)))
print('RMSE is {}'.format(rmse))
import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.yscale('log')
plt.show()