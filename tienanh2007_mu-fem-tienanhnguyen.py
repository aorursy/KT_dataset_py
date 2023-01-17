import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_json("../input/train.json")

test_df = pd.read_json("../input/test.json")
# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

y_train = np.array(train_df["is_iceberg"])

print("Xtrain:", X_train.shape)



# Test data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])

X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

print("Xtest:", X_test.shape)
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras import initializers

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from sklearn.model_selection import train_test_split

import os
def getModel():

    #Building the model

    gmodel=Sequential()

    #Conv Layer 1

    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 2)))

    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    gmodel.add(Dropout(0.2))



    #Conv Layer 2

    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    gmodel.add(Dropout(0.2))



    #Conv Layer 3

    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    gmodel.add(Dropout(0.2))



    #Conv Layer 4

    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    gmodel.add(Dropout(0.2))



    #Flatten the data for upcoming dense layers

    gmodel.add(Flatten())



    #Dense Layers

    gmodel.add(Dense(512))

    gmodel.add(Activation('relu'))

    gmodel.add(Dropout(0.2))



    #Dense Layer 2

    gmodel.add(Dense(256))

    gmodel.add(Activation('relu'))

    gmodel.add(Dropout(0.2))



    #Sigmoid Layer

    gmodel.add(Dense(1))

    gmodel.add(Activation('sigmoid'))



    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    gmodel.compile(loss='binary_crossentropy',

                  optimizer=mypotim,

                  metrics=['accuracy'])

    gmodel.summary()

    return gmodel

def get_callbacks(filepath, patience=2):

    es = EarlyStopping('val_loss', patience=patience, mode="min")

    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]

file_path = ".model_weights.hdf5"

callbacks = get_callbacks(filepath=file_path, patience=5)
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1, train_size=0.75)
model=getModel()

model.fit(X_train_cv, y_train_cv,

          batch_size=24,

          epochs=50,

          verbose=1,

          validation_data=(X_valid, y_valid))
model.load_weights(filepath=file_path)

predicted_test=model.predict_proba(X_test)
submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': predicted_test.flatten()})

submit_df.to_csv("./naive_submission.csv", index=False)