from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
housing_datasets = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing_datasets.data, housing_datasets.target)

x_train, x_validation, y_train, y_validation = train_test_split(x_train_full, y_train_full)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_validation = scaler.fit_transform(x_validation)

x_test = scaler.fit_transform(x_test)
model = tf.keras.Sequential([tf.keras.layers.Dense(30, activation = 'relu', input_shape = x_train.shape[1:]),

                            tf.keras.layers.Dense(1)])

model.compile(loss = tf.losses.mean_squared_error,

             optimizer = 'sgd')
hostory = model.fit(x_train, y_train, epochs=20,

                   validation_data=(x_validation, y_validation))
mean_squared_error_test = model.evaluate(x_test, y_test)

x_new = x_test[:3] # New instance

y_pred = model.predict(x_new)

y_pred
input_a = tf.keras.layers.Input(shape = [5], name = 'wide_input')

input_b = tf.keras.layers.Input(shape = [6], name = 'deep_input')

hidden1 = tf.keras.layers.Dense(30, activation = 'relu')(input_b)

hidden2 = tf.keras.layers.Dense(30, activation = 'relu')(hidden1)

concat = tf.keras.layers.concatenate([input_a, hidden2])

output = tf.keras.layers.Dense(1, name = 'output')(concat)
model = tf.keras.Model(inputs = [input_a, input_b], outputs = [output])
model.compile(loss = tf.losses.mean_squared_error,

             optimizer = tf.keras.optimizers.SGD(lr = 1e-3))
x_train_a, x_train_b = x_train[:, :5], x_train[:, 2:]

x_validation_a, x_validation_b = x_validation[:, :5], x_validation[:, 2:]

x_test_a, x_test_b = x_test[:, :5], x_test[:, 2:]

x_new_a, x_new_b = x_test_a[:3], x_test_b[:3]
history = model.fit((x_train_a, x_train_b), y_train, epochs=20, validation_data=((x_validation_a, x_validation_b), y_validation))

mean_squared_error_test = model.evaluate((x_test_a, x_test_b), y_test)

y_pred = model.predict((x_new_a, x_new_b))