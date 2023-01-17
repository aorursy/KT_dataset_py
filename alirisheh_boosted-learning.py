import numpy as np

import datetime

from tensorflow.keras.datasets import fashion_mnist

import tensorflow.keras.backend as K 

(X_train, Y_train) , (X_test, Y_test) = fashion_mnist.load_data()
X_train.shape
X_train[0]
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28*28)

X_test = X_test.reshape(-1,28*28)

X_train.shape
import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()
model2 = tf.keras.models.Sequential()

model2.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))

model2.add(tf.keras.layers.Dropout(0.4))

model2.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model2.summary()
model.fit(X_train, Y_train, epochs=5)
model2.fit(X_train, Y_train, epochs=5)
model3 = tf.keras.models.Sequential()

model3.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))

model3.add(tf.keras.layers.Dropout(0.4))

model3.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model3.summary()
model3.fit(X_train, Y_train, epochs=5)
model.predict(np.array([X_test[1]]))[0]
y_hat = [np.argmax(model.predict(np.array([i,]))[0]) for i in X_train]

count = 0

for i in range(len(X_train)):

    if y_hat[i] != Y_train[i]:

        count += 1

print(count/len(X_train))
y_hat = [np.argmax(model2.predict(np.array([i,]))[0]) for i in X_train]

count = 0

for i in range(len(X_train)):

    if y_hat[i] != Y_train[i]:

        count += 1

print(count/len(X_train))
y_hat = [np.argmax(model3.predict(np.array([i,]))[0]) for i in X_train]

count = 0

for i in range(len(X_train)):

    if y_hat[i] != Y_train[i]:

        count += 1

print(count/len(X_train))
ans_1 = model.predict(X_train)

ans_2 = model2.predict(X_train)

ans_3 = model3.predict(X_train)
print(ans_1[0], ' ', ans_2[0])

print(np.dot(ans_1[0], ans_2[0]))
def boost_error(y_true, y_predict):

    return K.sum(K.abs(y_true - y_predict))
boost_model = tf.keras.models.Sequential()

boost_model.add(tf.keras.layers.Dense(units=1, input_shape=(3,)))

boost_model.summary()
boost_model.compile(optimizer='adam', loss='MSE', metrics=['MSE', boost_error])
X_boost = [[np.argmax(ans_1[i]), np.argmax(ans_2[i]), np.argmax(ans_3[i])] for i in range(len(X_train))]
X_boost = np.array(X_boost)
boost_model.fit(X_boost, Y_train, epochs=20)
dots = boost_model.predict(X_boost)
dots[0]
count = 0

for i in range(len(X_train)):

    if np.around(dots[i][0]) != Y_train[i]:

        count += 1

print(count/len(X_train))