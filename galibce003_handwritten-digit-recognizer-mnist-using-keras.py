import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

X_train[0]
X_train[0].shape
plt.matshow(X_train[0])
y_train[0]
X_train = X_train / 255
X_test = X_test / 255
X_train[0]

X_train.shape
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_train_flattened.shape
X_test.shape
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_test_flattened.shape
model = keras.Sequential()

model.add(Dense(10, input_shape = (784,), activation = 'sigmoid'))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(X_train_flattened, y_train, epochs = 5)
model.evaluate(X_test_flattened, y_test)[1]
y_pred = model.predict(X_test_flattened)
plt.matshow(X_test[0])
y_pred[0]
np.argmax(y_pred[0])
y_pred_labels = [np.argmax(i) for i in y_pred]
y_pred_labels[:5]
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_labels)

sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
model = keras.Sequential()

model.add(Dense(100, input_shape = (784,), activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(X_train_flattened, y_train, epochs = 5)
model.evaluate(X_test_flattened, y_test)[1]
y_pred = model.predict(X_test_flattened)
y_pred_labels = [np.argmax(i) for i in y_pred]
y_pred_labels[:5]
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_labels)

sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
model = keras.Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)
model.evaluate(X_test, y_test)[1]
model = keras.Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'RMSprop',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)

model.evaluate(X_test, y_test)[1]
model = keras.Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'SGD',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)

model.evaluate(X_test, y_test)[1]
model = keras.Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'Adadelta',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)

model.evaluate(X_test, y_test)[1]
model = keras.Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))

model.compile(optimizer = 'Nadam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)

model.evaluate(X_test, y_test)[1]