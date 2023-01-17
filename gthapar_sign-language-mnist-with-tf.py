# make sure tf is installed and available

!pip install --upgrade tensorflow
# print the version of tf installed

import tensorflow as tf

print(tf.__version__)
import numpy as np # linear algebra

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



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

    labels = []

    images = []

    with open(filename) as training_file:

        csv_reader = csv.reader(training_file)

        next(csv_reader)

        for row in csv_reader:

            labels.append(row[0])

            pixels = row[1:785]

            pixels_as_array = np.array_split(pixels, 28)

            images.append(pixels_as_array)

        labels = np.array(labels).astype('float')

        images = np.array(images).astype('float')

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
# In this section you will have to add another dimension to the data

# So, for example, if your array is (10000, 28, 28)

# You will need to make it (10000, 28, 28, 1)

# Hint: np.expand_dims

from tensorflow.keras.preprocessing.image import ImageDataGenerator



training_images = np.expand_dims(training_images, axis=3)

testing_images = np.expand_dims(testing_images, axis=3)



# Create an ImageDataGenerator and do Image Augmentation

train_datagen = ImageDataGenerator(

    rescale=1./255.,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



validation_datagen = ImageDataGenerator(

    rescale=1./255.

)

    

# Keep These

print(training_images.shape)

print(testing_images.shape)

    

# Their output should be:

# (27455, 28, 28, 1)

# (7172, 28, 28, 1)
# Define the model

# Use no more than 2 Conv2D and 2 MaxPooling2D

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dense(26, activation='softmax')

])



# Compile Model. 

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



len_training_images = len(training_images)

len_testing_images = len(testing_images)



train_flow = train_datagen.flow(training_images, training_labels, batch_size=32)

val_flow = validation_datagen.flow(testing_images, testing_labels, batch_size=32)



# Train the Model

history = model.fit_generator(

    train_flow,

    steps_per_epoch=len_training_images/32,

    epochs=15,

    validation_data=val_flow,

    validation_steps=len_testing_images/32

)



model.evaluate(testing_images, testing_labels)



# The output from model.evaluate should be close to:

[6.92426086682151, 0.56609035]
# Plot the chart for accuracy and loss on both training and validation



import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

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