import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2
X = []

y = []

IMG_SIZE = 150



DIR = "/kaggle/input/flowers-recognition/flowers/flowers"



folders = os.listdir(DIR)





folders
for i, file in enumerate(folders):

    filename = os.path.join(DIR, file)

    print("Folder {} started".format(file))

    try:

        for img in os.listdir(filename):

            path = os.path.join(filename, img)

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))



            X.append(np.array(img))

            y.append(i)

    except:

        print("File {} not read".format(path))

        

    print("Folder {} done".format(file))

    print("The folder {} is labeled as {}".format(file, i))
np.unique(y, return_counts=True)
X = np.array(X)

y = np.array(y)



print("X shape is {}".format(X.shape))

print("y shape is {}".format(y.shape))
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from random import sample



random_indexes = sample(range(1, 3500), 16)

print(random_indexes)



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



# Index for iterating over images





# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)







for i, img_index in enumerate(random_indexes):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.set_title(folders[y[img_index]])

  sp.axis('Off') # Don't show axes (or gridlines)



  plt.imshow(X[img_index])





plt.show()
from tensorflow.keras.utils import to_categorical



print("Before the categorical the shape of y is {}".format(y.shape))

y = to_categorical(y)

print("After the categorical the shape of y is {}".format(y.shape))

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)



print("There are {} training examples".format(X_train.shape[0]))

print("There are {} test examples".format(X_test.shape[0]))
import tensorflow as tf

import keras_preprocessing
training_datagen = keras_preprocessing.image.ImageDataGenerator(

      rescale = 1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



training_datagen.fit(X_train)



validation_datagen = keras_preprocessing.image.ImageDataGenerator(

      rescale = 1./255)



validation_datagen.fit(X_test)
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')

])



model.summary()



model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
epochs=50

batch_size=32



history = model.fit_generator(training_datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = 50, validation_data = validation_datagen.flow(X_test, y_test, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
predictions = model.predict(X_test)

prediction_digits = np.argmax(predictions, axis=1)



labels_pred = np.unique(prediction_digits, return_counts=True)

real_labels = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix



c_m = confusion_matrix(real_labels, prediction_digits)



c_m
import seaborn as sns

plt.figure(figsize = (10,10))

sns.heatmap(c_m,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = folders , yticklabels = folders)
from tensorflow.keras.applications import VGG16



base_model = VGG16(input_shape=(150, 150, 3),

                  include_top=False,

                  weights="imagenet")



print("Base Model Loaded")
for layer in base_model.layers:

    layer.trainable=False
base_model.summary()
last_layer = base_model.get_layer('block5_pool')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output


from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras import layers, Model



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense  (5, activation='softmax')(x)           



model = Model( base_model.input, x) 



model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])



model.summary()
history = model.fit_generator(training_datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = 50, validation_data = validation_datagen.flow(X_test, y_test, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
predictions = model.predict(X_test)

prediction_digits = np.argmax(predictions, axis=1)



labels_pred = np.unique(prediction_digits, return_counts=True)

real_labels = np.argmax(y_test, axis=1)
c_m = confusion_matrix(real_labels, prediction_digits)



c_m
plt.figure(figsize = (10,10))

sns.heatmap(c_m,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = folders , yticklabels = folders)


np.unique(real_labels, return_counts=True)