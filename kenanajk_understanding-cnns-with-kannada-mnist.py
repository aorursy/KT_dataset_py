import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential

from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D

from keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.utils.np_utils import to_categorical



from sklearn.model_selection import train_test_split
# Load the data

Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")



# Split the data

x = train.iloc[:,1:].values

y = train.iloc[:,0].values

y[:10]

x_test = test.drop('id', axis=1).iloc[:,:].values

x_dig = Dig_MNIST.drop('label', axis=1).iloc[:,:].values

y_dig = Dig_MNIST.label



# Reshape the data

x = x.reshape(x.shape[0], 28, 28, 1)

y = to_categorical(y, 10)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_dig = x_dig.reshape(x_dig.shape[0], 28, 28, 1)



# Split the data between train and test

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.10, random_state=42) 



# Artificially increase training set

train_datagen = ImageDataGenerator(rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)



valid_datagen = ImageDataGenerator(rescale=1./255.)
batch_size = 1024

num_classes = 10

epochs = 50

learning_rate = 0.001

model_name = 'k-mnist_trained_model.h5'



model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(256, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()



optimizer = RMSprop(lr=learning_rate)



model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=200,

                                            verbose=1,

                                            factor=0.2)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)



history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),

                              steps_per_epoch=100,

                              epochs=epochs,

                              validation_data=valid_datagen.flow(x_valid, y_valid),

                              validation_steps=50,

                              callbacks=[learning_rate_reduction, es])
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predictions = model.predict_classes(x_test/255.)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv("submission.csv", index=False)