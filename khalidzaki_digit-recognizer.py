import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D,Dense  , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
y_train[:10]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train.shape

X_test.shape

X_val.shape

import tensorflow as tf



training_images=X_train.reshape(37800, 28, 28, 1)

training_images=training_images / 255.0

test_images = X_test.reshape(28000, 28, 28, 1)

test_images=test_images/255.0

val_images = X_val.reshape(4200,28,28,1)

val_images = val_images/255.0



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (5,5), activation='relu',padding='same', input_shape=(28, 28, 1)),

  tf.keras.layers.Conv2D(32,(5,5), activation='relu',padding='same'),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout((0.25)),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)),

  tf.keras.layers.Dropout((0.25)),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dropout((0.5)),

  tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer = tf.keras.optimizers.Adam( epsilon=1e-08), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly shift images horizontally (fraction of total width)

train_datagen.fit(training_images)
history = model.fit_generator(train_datagen.flow(training_images,y_train,batch_size = 86),

                              epochs=35,

                              verbose=2,

                              steps_per_epoch=training_images.shape[0] // 86,

                              validation_data=(val_images,y_val),

                              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3,

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)])
%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.figure()
predictions = model.predict_classes(test_images, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
submissions.head()
