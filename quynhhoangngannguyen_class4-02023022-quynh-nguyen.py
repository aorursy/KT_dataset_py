# https://playground.tensorflow.org

# upload pima-indians-diabetes.data.csv PATH ../input/pima-indians-diabetes.data.csv
import os

os.listdir('../input')
import tensorflow

print(tensorflow.__version__)
# https://github.com/hunkim/DeepLearningZeroToAll/tree/master/tf2
import numpy as np

import tensorflow as tf



x_train = [1, 2, 3, 4]

y_train = [0, -1, -2, -3]



tf.model = tf.keras.Sequential()

# units == output shape, input_dim == input shape

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))



sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate

tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2



# prints summary of the model to the terminal

tf.model.summary()



# fit() executes training

tf.model.fit(x_train, y_train, epochs=200)



# predict() returns predicted value

y_predict = tf.model.predict(np.array([5, 4]))

print(y_predict)
# Lab 3 Minimizing Cost

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt



x_train = [1, 2, 3, 4]

y_train = [0, -1, -2, -3]



tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))



sgd = tf.keras.optimizers.SGD(lr=0.1)

tf.model.compile(loss='mse', optimizer=sgd)



tf.model.summary()



# fit() trains the model and returns history of train

history = tf.model.fit(x_train, y_train, epochs=100)



y_predict = tf.model.predict(np.array([5, 4]))

print(y_predict)



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Lab 4 Multi-variable linear regression

import tensorflow as tf

import numpy as np



x_data = [[73., 80., 75.],

          [93., 88., 93.],

          [89., 91., 90.],

          [96., 98., 100.],

          [73., 66., 70.]]

y_data = [[152.],

          [185.],

          [180.],

          [196.],

          [142.]]



tf.model = tf.keras.Sequential()



tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  # input_dim=3 gives multi-variable regression

tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default

# advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6



tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))

tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=100)



y_predict = tf.model.predict(np.array([[72., 93., 90.]]))

print(y_predict)
# Lab 5 Logistic Regression Classifier

import tensorflow as tf

import numpy as np



xy = np.loadtxt('../input/pima-indians-diabetes.data.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]

y_data = xy[:, [-1]]



print(x_data.shape, y_data.shape)



tf.model = tf.keras.Sequential()

# multi-variable, x_data.shape[1] == feature counts == 8 in this case

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=x_data.shape[1], activation='sigmoid'))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01),  metrics=['accuracy'])

tf.model.summary()



history = tf.model.fit(x_data, y_data, epochs=500)



# accuracy!

print("Accuracy: {0}".format(history.history['accuracy'][-1]))



# predict a single data point

y_predict = tf.model.predict([[0.176471, 0.155779, 0, 0, 0, 0.052161, -0.952178, -0.733333]])

print("Predict: {0}".format(y_predict))



# evaluating model

evaluate = tf.model.evaluate(x_data, y_data)

print("loss: {0}, accuracy: {1}".format(evaluate[0], evaluate[1]))
# Lab 6 Softmax Classifier

import tensorflow as tf

import numpy as np



x_raw = [[1, 2, 1, 1],

          [2, 1, 3, 2],

          [3, 1, 3, 4],

          [4, 1, 5, 5],

          [1, 7, 5, 5],

          [1, 2, 5, 6],

          [1, 6, 6, 6],

          [1, 7, 7, 7]]

y_raw = [[0, 0, 1],

          [0, 0, 1],

          [0, 0, 1],

          [0, 1, 0],

          [0, 1, 0],

          [0, 1, 0],

          [1, 0, 0],

          [1, 0, 0]]



x_data = np.array(x_raw, dtype=np.float32)

y_data = np.array(y_raw, dtype=np.float32)



nb_classes = 3



tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(input_dim=4, units=nb_classes, use_bias=True))  # use_bias is True, by default



# use softmax activations: softmax = exp(logits) / reduce_sum(exp(logits), dim)

tf.model.add(tf.keras.layers.Activation('softmax'))



# use loss == categorical_crossentropy

tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

tf.model.summary()



history = tf.model.fit(x_data, y_data, epochs=2000)



print('--------------')

# Testing & One-hot encoding

a = tf.model.predict(np.array([[1, 11, 7, 9]]))

print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))



print('--------------')

b = tf.model.predict(np.array([[1, 3, 4, 3]]))

print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))



print('--------------')

# or use argmax embedded method, predict_classes

c = tf.model.predict(np.array([[1, 1, 0, 1]]))

c_onehot = tf.model.predict_classes(np.array([[1, 1, 0, 1]]))

print(c, c_onehot)



print('--------------')

all = tf.model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))

all_onehot = tf.model.predict_classes(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))

print(all, all_onehot)
# Lab 9 XOR

import tensorflow as tf

import numpy as np



x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)



tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01),  metrics=['accuracy'])

tf.model.summary()



history = tf.model.fit(x_data, y_data, epochs=1000)



predictions = tf.model.predict(x_data)

print('Prediction: \n', predictions)



score = tf.model.evaluate(x_data, y_data)

print('Accuracy: ', score[1])
# Lab 9 XOR

import tensorflow as tf

import numpy as np



x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)



tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2))

tf.model.add(tf.keras.layers.Activation('sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))

tf.model.add(tf.keras.layers.Activation('sigmoid'))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(lr=0.1),  metrics=['accuracy'])

tf.model.summary()



history = tf.model.fit(x_data, y_data, epochs=10000)



predictions = tf.model.predict(x_data)

print('Prediction: \n', predictions)



score = tf.model.evaluate(x_data, y_data)

print('Accuracy: ', score[1])
# Lab 9 XOR

# 9-3 deep and wide

import tensorflow as tf

import numpy as np



x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)



tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=10, input_dim=2, activation='sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# SGD not working very well due to vanishing gradient problem, switched to Adam for now

# or you may use activation='relu', study chapter 10 to know more on vanishing gradient problem.

tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1), metrics=['accuracy'])

tf.model.summary()



history = tf.model.fit(x_data, y_data, epochs=5000)



predictions = tf.model.predict(x_data)

print('Prediction: \n', predictions)



score = tf.model.evaluate(x_data, y_data)

print('Accuracy: ', score[1])