import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

np.random.seed(123)



X = np.random.rand(1000, 4)

X[:500, [0, 1]] *= 0.1

X[500:, [2, 3]] *= 0.1

display(sns.heatmap(X))



y = np.zeros(1000)

y[500:] = 1.0



X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123)

cy_train = np_utils.to_categorical(y_train)

cy_valid = np_utils.to_categorical(y_valid)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(units=3, activation='relu', input_shape=(4,), use_bias=True))

model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])

print(model.summary())



for w in model.get_weights():

    print(w)
history = model.fit(X_train, cy_train, validation_data=(X_valid, cy_valid), batch_size=16, epochs=10)



for w in model.get_weights():

    print(w)
import numpy as np

import matplotlib.pylab as plt



def relu(x):

    return np.maximum(0, x)



# test

print(relu(np.array([-.3, -.1, .2, .4])))



# graph

x = np.arange(-5.0, 5.0, 0.1)

y = relu(x)

plt.plot(x, y)

plt.show()
import numpy as np

import matplotlib.pylab as plt



def softmax(a):

    exp_a = np.exp(a)

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a

    return y



# test

print(softmax(np.array([-.3, -.1, .2, .4])))



# graph

x = np.arange(-5, 5, 0.1)

y = softmax(x)

plt.plot(x, y)

plt.show()
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], color='blue')

plt.plot(history.history['val_loss'], color='orange')

plt.show()
X_test = np.array([[.01, .01, .98, .85], 

                   [.98, .57, .01, .02],])

print(model.predict(X_test))
weights = model.get_weights()

weights
unit = np.dot(X_test, weights[0])

print(unit)

unit += weights[1]

print(unit)

unit = relu(unit)

print(unit)

unit = np.dot(unit, weights[2])

print(unit)

unit += weights[3]

print(unit)

unit = softmax(unit)

print(unit)

unit /= unit.sum(axis=1)[:, np.newaxis]

unit
X_test = np.array([[.01, .01, .98, .85], 

                   [.98, .57, .01, .02],])

model.predict(X_test)