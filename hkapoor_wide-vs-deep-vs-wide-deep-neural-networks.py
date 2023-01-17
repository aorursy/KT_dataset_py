from tensorflow import keras
mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full.shape
X_test.shape
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28, 28]))

model.add(keras.layers.Dense(500, activation="relu"))

model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)
mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28, 28]))

model.add(keras.layers.Dense(200, activation="relu"))

model.add(keras.layers.Dense(200, activation="relu"))

model.add(keras.layers.Dense(200, activation="relu"))

model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)
mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full = X_train_full.reshape(60000, 784) / 255.0

X_test = X_test.reshape(10000, 784) / 255.0
input_ = keras.layers.Input(shape=(784,))

hidden1 = keras.layers.Dense(200, activation="relu")(input_)

hidden2 = keras.layers.Dense(200, activation="relu")(hidden1)

hidden3 = keras.layers.Dense(200, activation="relu")(hidden2)

concat = keras.layers.Concatenate()([input_, hidden3])

output = keras.layers.Dense(10, activation="softmax")(concat)

model = keras.Model(inputs=[input_], outputs=[output])
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train_full, y_train_full, epochs=10, validation_split=0.1)
model.evaluate(X_test, y_test)