from pandas import read_csv

import numpy as np

from keras.models import Sequential

from keras.layers import Dense
import pandas as pd

housing_df = pd.read_csv("../input/housing.csv")



dataset = housing_df.values
X = dataset[:,0:13]

Y = dataset[:,13]



def mlp_model():

    model = Sequential()

    model.add(Dense(10, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(6, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



model = mlp_model()

model.summary()
history = model.fit(X, Y, epochs = 100, batch_size = 5, validation_split=0.2)
import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model accuracy')

plt.ylabel('MSE')

plt.xlabel('Epoch')

plt.legend(['Train MSE', 'Test MSE'], loc='upper left')

plt.show()
my_new_data = np.asarray((0.08829,12.50,7.870,0,0.5240,6.0120,66.60,5.5605,5,311.0,15.20,395.60,12.43))

my_new_data = my_new_data.reshape((1,13))

price_prediction = model.predict(my_new_data)



print(price_prediction)