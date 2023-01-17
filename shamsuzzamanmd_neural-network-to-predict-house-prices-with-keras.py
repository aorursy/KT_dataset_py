import pandas as pd



# data reading

df = pd.read_csv("../input/housepricedata.csv")
# data set view

df.head()
# columns

df.columns
# brief info

df.info()
# data frame to arrays for computational analysis

dataset = df.values

dataset
# Let´s define X and Y features



X = dataset[:,0:10]

Y = dataset[:,10]
# sklearn import

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)

#X_scale
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test,Y_val_and_test, test_size=0.5)
# Shape of the data split

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
# keras

import tensorflow as tf

from tensorflow import keras





from keras.models import Sequential

from keras.layers import Dense
model = tf.keras.Sequential([

    keras.layers.Dense(32, activation='relu', input_shape=(10,)),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid'),

])
# configuring the model

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
# Training the model

hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
# Evaluating the model

model.evaluate(X_test, Y_test)[1] # index 1 beacuse this function returns at first index "loss" and at 2nd "accuracy"
import matplotlib.pyplot as plt

%matplotlib inline
# Training loss Vs. Validation loss

plt.plot(hist.history["loss"])

plt.plot(hist.history["val_loss"])

plt.title("Model loss")

plt.ylabel("Loss")

plt.xlabel("Epoch")

plt.legend(["Train", "Val"], loc="upper right")

plt.show()
# Training accuracy and Validation accuracy

plt.plot(hist.history["accuracy"])

plt.plot(hist.history["val_accuracy"])

plt.title("Model accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Val"], loc="lower right")

plt.show()
model_2=tf.keras.Sequential([keras.layers.Dense(1000, activation="relu", input_shape=(10,)),

                            keras.layers.Dense(1000, activation="relu"),

                            keras.layers.Dense(1000, activation="relu"),

                            keras.layers.Dense(1000, activation="relu"),

                            keras.layers.Dense(1, activation="sigmoid")])



model_2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



hist_2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
# loss Vs. val_loss

plt.plot(hist_2.history['loss'])

plt.plot(hist_2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
# Accuracy Vs. Val_accuracy

plt.plot(hist_2.history["accuracy"])

plt.plot(hist_2.history["val_accuracy"])

plt.title("Model accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Val"], loc="lower right")

plt.show()
from keras.layers import Dropout

from keras import regularizers
model_3 = tf.keras.Sequential()

model_3.add(keras.layers.Dense(1000,activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)))

model_3.add(keras.layers.Dropout(0.3))

model_3.add(keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model_3.add(keras.layers.Dropout(0.3))

model_3.add(keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model_3.add(keras.layers.Dropout(0.3))

model_3.add(keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model_3.add(keras.layers.Dropout(0.3))

model_3.add(keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
# Compiling and Fitting



model_3.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

hist_3 = model_3.fit(X_train, Y_train,

          batch_size=32, epochs=100,

          validation_data=(X_val, Y_val))
# Plot of loss Vs. val_loss



plt.plot(hist_3.history['loss'])

plt.plot(hist_3.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.ylim(top=1.2, bottom=0)

plt.show()
# Accuracy Vs. val_accuracy



plt.plot(hist_3.history['accuracy'])

plt.plot(hist_3.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()