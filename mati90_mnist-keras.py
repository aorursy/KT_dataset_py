import pandas as pd

import numpy as np



source = '/kaggle/input/digit-recognizer/{file}'



train_df = pd.read_csv(source.format(file='train.csv'))

test_df = pd.read_csv(source.format(file='test.csv'))



original_test_X = test_df.values



original_train_X = train_df.iloc[:, 1:].values

original_train_y = train_df.iloc[:, :1].values
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler





X = np.concatenate([original_train_X, original_test_X])



# Scale data

scaler = MinMaxScaler()

X = scaler.fit_transform(X)



X_train, X_test = np.split(X, [42000])

y_train = to_categorical(original_train_y)



X_train = np.reshape(X_train, (X_train.shape[0],) + (28,28, 1))

print(f"Data size X: {X_train.shape}, y: {y_train.shape}")



X_test = np.reshape(X_test, (X_test.shape[0],) + (28,28, 1))

print(f"Data size X: {X_test.shape}")
# Prepare model architecture

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.callbacks import ReduceLROnPlateau





# Prepare model architecture

model = tf.keras.Sequential([

    layers.Conv2D(32, kernel_size=3,padding='same',activation='relu', input_shape=(28,28,1)),

    layers.BatchNormalization(),

    layers.Conv2D(32, kernel_size=3,padding='same',activation='relu', input_shape=(28,28,1)),

    layers.BatchNormalization(),

    layers.MaxPool2D(),

    layers.BatchNormalization(),

    layers.Dropout(0.35),



    layers.Conv2D(48, kernel_size=3,padding='same',activation='relu'),

    layers.BatchNormalization(),

    layers.Conv2D(48, kernel_size=3,padding='same',activation='relu'),

    layers.BatchNormalization(),

    layers.MaxPool2D(),

    layers.BatchNormalization(),

    layers.Dropout(0.35),    



    layers.Conv2D(64, kernel_size=3,padding='same',activation='relu'),

    layers.BatchNormalization(),

    layers.Conv2D(64, kernel_size=3,padding='same',activation='relu'),

    layers.BatchNormalization(),

    layers.MaxPool2D(),

    layers.BatchNormalization(),

    layers.Dropout(0.35),



    layers.Flatten(),

    layers.Dense(units=128, activation='relu'),

    layers.BatchNormalization(),

    layers.Dropout(0.35),



    layers.Dense(units=64, activation='relu'),

    layers.BatchNormalization(),

    layers.Dropout(0.35),



    layers.Dense(10, activation='softmax')

])



callbacks_reduce_learning_rate = [

    ReduceLROnPlateau(monitor='loss', factor=0.6, verbose=1, patience=2, min_lr=0.000000001)

]
from tensorflow.keras.models import clone_model

from sklearn.model_selection import train_test_split





X_cross_train, X_cross_test, y_cross_train, y_cross_test = train_test_split(X_train, y_train, train_size=30000)

epochs = 1

validation_model = clone_model(model)

validation_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

validation_model.fit(X_cross_train, y_cross_train, epochs=epochs, validation_data=(X_cross_test, y_cross_test), callbacks=callbacks_reduce_learning_rate)
epochs = 50

train_model = clone_model(model)

train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



train_model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks_reduce_learning_rate)
predict_matrix = train_model.predict(X_test)

predict = np.argmax(predict_matrix,axis = 1)

predict = pd.Series(predict, name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predict],axis = 1)

submission.to_csv("submission.csv",index=False)