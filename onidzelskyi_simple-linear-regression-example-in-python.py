# import libraries for plotting and array initialization

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline





# define data

X = np.arange(1, 4,dtype=np.float64)  # create array from 1 to 3 inclusive (both 1 and 3 included) with step 1



f = lambda x: 3 * x + 2

Y = f(X)





# Plot data

plt.plot(X, Y, 'ro')
# Basic math library in Python

import numpy as np





# Add bias column to the X values 

X = np.c_[np.ones(X.shape[0]).T, X.T]



# Formula (2) in action

W_true = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)



# Print out results

# Value, calculated by normal equation is ground truth

print(W_true)
# Firstly, let's define our weights we need to estimate 

# and initialize it by zeros

# its dimentsion (# of weights) equals #features (second dimension of X)

W = np.zeros(shape=(X.shape[1], 1))



# And set learning rate to 0.01 

# (usually it vary form 0 to 1).

# The close to 0 means slowly learning

# close to 1 means faster learning

# Some algorithms automatically adjust learning rate 

# starting from bigger value at the start of learning

# and slowly decreasing it when learning come to end

# You can play with its value to compare results

alpha = .01



roc = []



# We took 50 iterations 

# #iterations can vary depending on task

# usually it lays between 50 - 1000 iterations

# depending on the network' complexity

# in our case we train simple linear regression

# so, 50 iterations enough

epochs = 50



# Other approach is to choose convergence limit

# and stop iterate when estimated value is close enough to the ground true

for i in range(epochs):

    # Firstly, calculate estimated value from formula (1.3)

    Y_hat = X.dot(W).T

    # Then, using MSE calc cost funciton, which estimate how accurately we calc predicted value

    cost = 2 * np.sum((Y_hat - Y) ** 2) / len(X)

    # Save our divergence for displaying our ROC curve

    roc.append(cost)

    # Calc differential (most complicated and unloved part of this algorithm)

    dw = np.expand_dims(2 * np.sum(((Y_hat - Y)).dot(X), axis=0) / len(X), axis=0)

    # Adjust weights. Here we perform main task

    W -= alpha * dw.T



# Print estimated weights    

print(W)    
import matplotlib.pyplot as plt

%matplotlib inline





plt.plot(roc)
# Our formula (1.3)

y = lambda x, w: X.dot(w).T.squeeze()



plt.plot(X[:, 1], Y, 'ro', label='Samples')

plt.plot(X[:, 1], y(X, W), linestyle='solid', label='Predicted')

plt.plot(X[:, 1], y(X, W_true), linestyle='dashed', label='Ground truth')

plt.legend()
import tensorflow as tf



model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')





history = model.fit(X[:, 1], Y, epochs=50, verbose=0)

plt.plot(history.history['loss'], label='keras SGD')

plt.plot(roc, label='manual gradient descent')

plt.legend()

model.get_weights()
X = np.arange(1, 17, dtype=np.float64)  # numbers from 1 to 16 inclusive 

Y = f(X)



model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(X, Y, epochs=50, verbose=0)



plt.plot(history.history['loss'], )
n = 18

X = np.arange(1, n, dtype=np.float64)  # numbers from 1 to 17 inclusive 

Y = f(X)



model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(X, Y, epochs=50, verbose=0)



plt.plot(history.history['loss'], )
# #Samples, you can change this value to make sure it is not hurt model' convergence.

n = 18



X = np.arange(1, n, dtype=np.float64)  # numbers from 1 to (n-1) inclusive 



# Normalization / feature scaling

# Most effective normalization for SGD

X_norm = (X - np.mean(X)) / np.std(X)



Y = f(X)



model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model. For more accurate results try to increase #epochs, say to 500

history = model.fit(X_norm, Y, epochs=50, verbose=0)



# Let's calc ground truth value for input value 19 using by f() lambda function

ground_truth = f(19)

# In contrast of line above, for predicted value we must normalize input data firstly

predicted =  model.predict([(19-np.mean(X))/np.std(X)])[0][0]

print('Ground truth: {}, predicted: {}'.format(ground_truth, predicted))



plt.plot(history.history['loss'])