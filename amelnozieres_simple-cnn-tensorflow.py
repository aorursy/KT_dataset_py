# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
def get_data(filename):

    with open(filename) as training_file:

      # Your code starts here

        my_arr = np.loadtxt(filename, delimiter=',', skiprows=1)

        # get label & image arrays

        labels = my_arr[:,0].astype('int')

        images = my_arr[:,1:]

        # reshape image from 784 to (28, 28)

        images = images.astype('float').reshape(images.shape[0], 28, 28)

        # just in case to avoid memory problem

        my_arr = None

      # Your code ends here

    return images, labels





training_images, training_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

testing_images, testing_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')



# Keep these

print(training_images.shape)

print(training_labels.shape)

print(testing_images.shape)

print(testing_labels.shape)



# Their output should be:

# (27455, 28, 28)

# (27455,)

# (7172, 28, 28)

# (7172,)
training_images = np.expand_dims(training_images, axis =3) 

testing_images = np.expand_dims(testing_images, axis = 3)



# Create an ImageDataGenerator and do Image Augmentation

train_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    #horizontal_flip=True,

    fill_mode='nearest')





validation_datagen = ImageDataGenerator(rescale = 1./255)

    



print(training_images.shape)

print(testing_images.shape)

    

# Their output should be:

# (27455, 28, 28, 1)

# (7172, 28, 28, 1)
model = tf.keras.models.Sequential([

    # Your Code Here

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(25, activation='softmax')

])



model.summary()



# Compile Model. 

model.compile(loss = 'sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam( learning_rate=0.001), metrics=['accuracy'])



# Train the Model

# training_labels_bin = tf.keras.utils.to_categorical(training_labels)

# testing_labels_bin = tf.keras.utils.to_categorical(testing_labels)



history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=10),

                              epochs=5, 

                              steps_per_epoch = len(training_images) // 10,

                              validation_data = validation_datagen.flow(testing_images, testing_labels))# Your Code Here (set 'epochs' = 2))



model.evaluate(testing_images, testing_labels, verbose=0)
# Plot the chart for accuracy and loss on both training and validation

%matplotlib inline

import matplotlib.pyplot as plt

acc = history.history['accuracy'] # Your Code Here

val_acc = history.history['val_accuracy'] # Your Code Here

loss =  history.history['loss']# Your Code Here

val_loss = history.history['val_loss']# Your Code Here



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()