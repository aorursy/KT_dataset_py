# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import csv

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
def get_data(filename):

  # You will need to write code that will read the file passed

  # into this function. The first line contains the column headers

  # so you should ignore it

  # Each successive line contians 785 comma separated values between 0 and 255

  # The first value is the label

  # The rest are the pixel values for that picture

  # The function will return 2 np.array types. One with all the labels

  # One with all the images

  #

  # Tips: 

  # If you read a full line (as 'row') then row[0] has the label

  # and row[1:785] has the 784 pixel values

  # Take a look at np.array_split to turn the 784 pixels into 28x28

  # You are reading in strings, but need the values to be floats

  # Check out np.array().astype for a conversion

    with open(filename) as training_file:

      csv_reader =csv.reader(training_file, delimiter=',')

      first_line = True

      temp_images = []

      temp_labels = []

      for row in csv_reader:

        if first_line:

          first_line = False

        else:

          temp_labels.append(row[0])

          images_data = row[1:785]

          images_data_as_array = np.array_split(images_data, 28)

          temp_images.append(images_data_as_array)

          

      images = np.array(temp_images).astype('float')

      labels = np.array(temp_labels).astype('float')

      

    return images, labels
training_images, training_labels = get_data('../input/sign_mnist_train.csv')

testing_images, testing_labels = get_data('../input/sign_mnist_test.csv')



# Keep these

print(training_images.shape)

print(training_labels.shape)

print(testing_images.shape)

print(testing_labels.shape)

# In this section you will have to add another dimension to the data

# So, for example, if your array is (10000, 28, 28)

# You will need to make it (10000, 28, 28, 1)

# Hint: np.expand_dims



training_images = np.expand_dims(training_images,axis=3)

testing_images = np.expand_dims(testing_images,axis=3)



# Create an ImageDataGenerator and do Image Augmentation

train_datagen = ImageDataGenerator(

    rescale=1.0/255.0,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

    )



validation_datagen = ImageDataGenerator(

    rescale=1.0/255.0)

    

# Keep These

print(training_images.shape)

print(testing_images.shape)

    
class myCallback(tf.keras.callbacks.Callback):

  def on_eponch_end(self, epoch, logs={}):

    if(logs.get('acc') > 0.999):

      print("\n Reached 99.9% accuracy, so cancelling training. ")

      self.model.stop_training = True

  
# Define the model

# Use no more than 2 Conv2D and 2 MaxPooling2D

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(26, activation='softmax')

    

])



# Compile Model. 

model.compile(optimizer = 'adam',

              loss ='sparse_categorical_crossentropy',

              metrics = ['acc'])



train_generator = train_datagen.flow(

    training_images,

    training_labels,

    batch_size=32)

validation_generator = validation_datagen.flow(

    testing_images,

    testing_labels,

    batch_size=32)



callbacks = myCallback()



# Train the Model

history = model.fit_generator(train_generator,

                              steps_per_epoch = len(training_images)/32,

                              epochs = 50,

                              validation_data = validation_generator,

                              validation_steps = len(testing_images)/32,

                              callbacks = [callbacks])

model.evaluate(testing_images, testing_labels)
# Plot the chart for accuracy and loss on both training and validation



import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



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