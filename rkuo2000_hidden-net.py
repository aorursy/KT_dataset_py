# Import libraries

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Dense
# Data Generator

# create a matrix of row:300 col:1, value = -1~1

x_data = np.linspace(-1,1,300)[:, np.newaxis]

noise = np.random.normal(0, 0.05, x_data.shape)



# generate Y with noise for training model

y_data = np.square(x_data) - 0.5 + noise
# Build Model

model = keras.models.Sequential()



model.add(Dense(10, input_dim=1, activation='relu')) # add a hidden layer with 10 neurons 

model.add(Dense(1, activation=None))



model.summary()
# Compile Model

# Optimizer : SGD, RMSpro, Adagrad, Adaelta, Adam, Adamax, Nadam

model.compile(optimizer='sgd', loss='mse')
# Train Model

model.fit(x_data, y_data, batch_size=50, epochs=1000)
# Prediction : input X to predict Y

y_pred = model.predict(x_data)
# Draw Results

y_ideal= np.square(x_data) - 0.5          # use x_data to generate ideal y



plt.scatter(x_data, y_data) # plot training data (x_data ,y_data)

plt.plot(x_data, y_ideal, 'yellow', lw=2) # plot (x_data, y_ideal)

plt.plot(x_data, y_pred, 'red', lw=1)     # plot (x_data, y_pred)

plt.show()                                # show plot