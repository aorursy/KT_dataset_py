import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

# %matplotlib inline

dataset_train = pd.read_csv('../input/digit-recognizer/train.csv')

dataset_test = pd.read_csv('../input/digit-recognizer/test.csv')

x_train = dataset_train.iloc[:,1:].values.astype("float32") / 255

y_train = dataset_train.iloc[:,0].values

x_test = dataset_test.values.astype("float32") / 255



x_val = x_train[-10000:]

y_val = y_train[-10000:]

x_train = x_train[:-10000]

y_train = y_train[:-10000]

dataset_train.describe()

print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)
visualization = x_train.reshape(x_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(3, 3, i+1)

    plt.imshow(visualization[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit(X_train)

X_train_norm = norm.transform(X_train)

X_test_norm = norm.transform(X_test)



# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler().fit(X_train)

# X_train_scaled = scaler.transform(X_train)

# X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)
pd.DataFrame(X_train_scaled[:,100:120]).describe()
from sklearn.preprocessing import OneHotEncoder

y_encoder = OneHotEncoder()

y_encoder.fit(y_train.reshape(-1, 1))



print(y_encoder.categories_)

print(y_encoder.inverse_transform(([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])))
import tensorflow as tf
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(64, activation='relu'))

ann.add(tf.keras.layers.Dense(64, activation='relu'))

ann.add(tf.keras.layers.Dense(10, activation='softmax'))
ann.compile(

    optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer

    # Loss function to minimize

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    # List of metrics to monitor

    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],

)
ann.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))
y_pred = ann.predict(x_test)



y_pred = y_encoder.inverse_transform(y_pred>0.5)

print(y_pred[1])

print(y_pred[2])

print(y_pred[3])

print(y_pred[4])

pd.DataFrame(y_pred).describe()

answer_raw = {'ImageId': np.arange(1, len(y_pred) + 1), 'Label': y_pred[:, 0]}

print(answer_raw['Label'].shape)

print(answer_raw['ImageId'].shape)

print(answer_raw)

answer = pd.DataFrame(data=answer_raw)

print(answer.head())
answer.to_csv('submission.csv', index=False)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder





inputs = keras.Input(shape=(784,), name="digits")

x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

x = layers.Dense(64, activation="relu", name="dense_2")(x)

outputs = layers.Dense(10, activation="softmax", name="predictions")(x)



model = keras.Model(inputs=inputs, outputs=outputs)





dataset_train = pd.read_csv('../input/digit-recognizer/train.csv')

dataset_test = pd.read_csv('../input/digit-recognizer/test.csv')

print(dataset_train.shape)

print(dataset_test.shape)



x_train = dataset_train.iloc[:,1:].values.astype("float32") / 255

y_train = dataset_train.iloc[:,0].values

x_test = dataset_test.values.astype("float32") / 255



# # Reserve 10,000 samples for validation

x_val = x_train[-10000:]

y_val = y_train[-10000:]

x_train = x_train[:-10000]

y_train = y_train[:-10000]

print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)
# y_encoder = OneHotEncoder()

# y_encoder.fit(y_train.reshape(-1, 1))

# model.compile(

#     optimizer=keras.optimizers.RMSprop(),  # Optimizer

#     # Loss function to minimize

#     loss=keras.losses.SparseCategoricalCrossentropy(),

#     # List of metrics to monitor

#     metrics=[keras.metrics.SparseCategoricalAccuracy()],

# )

# print("Fit model on training data")

# history = model.fit(

#     x_train,

#     y_train,

#     batch_size=64,

#     epochs=2,

#     # We pass some validation for

#     # monitoring validation loss and metrics

#     # at the end of each epoch

#     validation_data=(x_val, y_val),

# )

# history.history
# y_pred = model.predict(x_test)



# y_pred = y_encoder.inverse_transform(y_pred>0.5)

# print(y_pred[1])

# print(y_pred[2])

# print(y_pred[3])

# print(y_pred[4])

# pd.DataFrame(y_pred).describe()


# answer_raw = {'ImageId': np.arange(1, len(y_pred) + 1), 'Label': y_pred[:, 0]}

# print(answer_raw['Label'].shape)

# print(answer_raw['ImageId'].shape)

# print(answer_raw)

# answer = pd.DataFrame(data=answer_raw)

# answer.describe()
# answer.to_csv('submission.csv', index=False)