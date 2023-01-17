from IPython.display import Image

Image("/kaggle/input/sign-language-mnist/amer_sign2.png")
# imports

import csv

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.utils import to_categorical
def get_data(filepath):

    

    with open(filepath) as f:

        

        csv_reader = csv.reader(f, delimiter=',')

        

        # set the first line var

        first_line = True

        

        # this will hold the images and labels

        features = []

        labels = []

        

        # iterate over the file

        for row in csv_reader:

            

            # ignore the first line

            if first_line:

                first_line = False

                

            # for all the other lines

            else:

                

                # append the first val to label

                labels.append(row[0])

                

                # read in the remaining values for image data

                image_data = row[1:785]

                

                # convert it to 28x28

                image_array = np.array_split(image_data, 28)

                

                # append this array to features

                features.append(image_array)

                

        # convert to numpy arrays with dtype 'float'

        features = np.array(features).astype('float')

        labels = np.array(labels).astype('float')

        

    return features, labels





training_images, training_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

testing_images, testing_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')



print(f"Shape of training features: {training_images.shape}")

print(f"Shape of training labels: {training_labels.shape}")

print(f"Shape of test features: {testing_images.shape}")

print(f"Shape of test labels: {testing_labels.shape}")
# reshape features to be of (n_samples, img_rows, img_cols, n_channels)

training_images = np.expand_dims(training_images, axis=3)

testing_images = np.expand_dims(testing_images, axis=3)



print(f"Shape of training features: {training_images.shape}")

print(f"Shape of test features: {testing_images.shape}")
# Normalize the images

training_images /= 255.0

testing_images /= 255.0
# encode the label

training_labels=to_categorical(training_labels)

testing_labels=to_categorical(testing_labels)
# some global variables

IMG_CHANNELS=1

IMG_ROWS=28

IMG_COLS=28

NB_CLASSES=25



# create the model

model = Sequential()



# layer1 - input

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# layer2

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# layer3

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))



# layer4 - output

model.add(Dense(NB_CLASSES))

model.add(Activation('softmax'))



optimizer=Adam(learning_rate=0.0001)

model.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics=['acc'])



history = model.fit(training_images,training_labels,validation_data=(testing_images,testing_labels),epochs=30)
# plot the training and testing accuracy and loss



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