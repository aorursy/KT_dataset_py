# import libraries

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras as keras
# Generate Dataset

# use math formula to generate data

X = np.linspace(-1, 1, 200)

np.random.shuffle(X)

Y = 0.1* X + 0.3 + np.random.normal(0, 0.01, (200,))



# split data into two sets, one for training, the other for testing (for model to predict result)

X_train, Y_train = X[:160], Y[:160]

X_test,  Y_test  = X[160:], Y[160:]
# Build Model

model = tf.keras.Sequential()



# add one unit of fully connected layer (Dense)

model.add(keras.layers.Dense(units=1, input_dim=1))
# Compile Model：set optimizer and loss function (for regression)

# Optimizer = SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam 

model.compile(optimizer='sgd', loss='mse')
# Train Model 

model.fit(X_train, Y_train, batch_size = 40, epochs=300)
# Evaluate Model (use test dataset)

cost = model.evaluate(X_test, Y_test, batch_size=40)

print("test cost: {}".format(cost))

W, b = model.layers[0].get_weights()

print("weights = {}, biases= {}".format(W, b))
# Model Prediction (use X_test to get Y_pred from model)

Y_pred = model.predict(X_test)



plt.scatter(X_test, Y_test) # plot Test dataset (X_test, Y_test)

plt.plot(X_test, Y_pred)    # plot Prediction   (X_test, Y_Pred)

plt.show()                  # display the plot