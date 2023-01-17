import gc

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import layers, optimizers

from tensorflow.keras.callbacks import ReduceLROnPlateau



from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
X = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')



Y = X[['label']]

X = X.drop(["label"], axis=1)



X_train = X.values.reshape(X.shape[0], 28, 28, 1)

Y_train = tf.keras.utils.to_categorical(Y.values, 10)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')



X_train /= 255

X_test /= 255

X_val /= 255
datagen = ImageDataGenerator(

        rotation_range=10,

        zoom_range = 0.10,

        width_shift_range=0.1, 

        height_shift_range=0.1)



datagen.fit(X_train)
#Build model

model = tf.keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 padding='same',

                 input_shape=(28, 28, 1)))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (5, 5), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(layers.Dropout(0.2))

  

model.add(layers.Flatten())  

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.3))

  

model.add(layers.Dense(512, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.3))

  

model.add(layers.Dense(256, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.3))



model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(loss="categorical_crossentropy",

              optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),

              metrics=['accuracy'])



reduce_lr = ReduceLROnPlateau(monitor='val_acc',

                                factor=0.5,

                                patience=3,

                                min_lr=0.00001,

                                verbose=1)
model.fit(datagen.flow(X_train, Y_train, batch_size=128),

                    epochs=30,

                    validation_data=(X_val, Y_val),

                    verbose=1,

                    steps_per_epoch=X_train.shape[0] // 128,

                    callbacks=[reduce_lr])
Y_test = model.predict(X_test, batch_size=128, verbose=1)

Y_test = np.argmax(Y_test, axis=1)

frame = pd.DataFrame({'Label': Y_test})

frame.insert(0, 'ImageId', range(1, 1 + len(frame)))

frame.to_csv('output.csv', index=False)