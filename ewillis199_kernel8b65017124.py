#CS 4442 Assignment 3 

#Emma Henriksen-Willis 250833234

#Diana Boras 250868767



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from PIL import Image

import matplotlib.pyplot as plt

from keras.utils import to_categorical



#loading the data:

X = []

labels = []

alldata = '../input/notMNIST_small/notMNIST_small'

for directory in os.listdir(alldata):

    for image in os.listdir(alldata + '/' + directory):

        try:

            file_path = alldata + '/' + directory + '/' + image

            img = Image.open(file_path)

            img.load()

            img_data = np.asarray(img, dtype=np.int16)

            X.append(img_data)

            labels.append(directory) #the subdirectory is the class of the image

        except Exception as e:

            pass

print("Data loaded.")
#Data summary (note that 2 of the training samples aren't loading for some reason):

num_examples = len(X)

print("There are", num_examples, "data samples total.")

img_width = len(X[0]) 

img_height = len(X[0][0]) 

print("The size of the samples is", img_width, "by", img_height)

print("Example unprocessed images:")

plt.imshow(X[1], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()

print("This sample has label:", labels[1])

plt.imshow(X[8000], cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()

print("This sample has label:", labels[8000])
from keras.preprocessing.image import ImageDataGenerator



#Data processing:

X = np.asarray(X, dtype=np.int16).reshape(num_examples, img_width, img_height,1) #reshape is there to include 1D channel for monochrome image



#The labels are characters right now, need to convert them to integers:

int_labels = [ord(c)-ord('A') for c in labels]

#now convert labels to one hot vectors: 

one_hot_labels = to_categorical(int_labels, 10)

print("The new label sample format is", one_hot_labels[1])

one_hot_labels = np.asarray(one_hot_labels)



#Data augmentation:

#The images are already N by N and cropped to fit the characters exactly, so we don't want to further crop them. 

#Since we're using the smaller dataset, overfitting is a concern. This makes data augmentation critical.

#The characters lose meaning when flipped horizontally or vertically, so augmenting the training set that way won't help with training



#We can rotate to a small degree, stretch the images. Using the Keras ImageDataGenerator, we can do the following:

#rotation_range=25 - rotate by 25 degrees max

#width_shift_range=0.1 - stretch or compress width by up to 10%

#height_shift_range=0.1 - stretch or compress height by up to 10%

#shear_range=0.1 - stretch or compress diagonally by up to 10% 



aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,

                         height_shift_range=0.1, shear_range=0.2, 

                         horizontal_flip=False, fill_mode="nearest")



#This augmentation won't be run until training, when it will be applied to the training set as it is loaded.
#split data into training and validation set

from numpy.random import uniform



training_fraction = 0.85 #aiming for approx. 15915 training samples and 2809 validation samples

val_X = []

val_Y = []

train_X = []

train_Y = []



for i,sample in enumerate(X):

    rand = uniform()

    if rand < training_fraction:

        train_X.append(sample)

        train_Y.append(one_hot_labels[i])

    else:

        val_X.append(sample)

        val_Y.append(one_hot_labels[i])

        

val_X = np.asarray(val_X, dtype=np.int16)

train_X = np.asarray(train_X, dtype=np.int16)

val_Y = np.asarray(val_Y, dtype=np.int16)

train_Y = np.asarray(train_Y, dtype=np.int16)



#normalize the X inputs by dividing by 255 (max pixel value) to minimize the variation bewteen

#training samples

train_X = train_X.astype('float32')

val_X = val_X.astype('float32')

train_X /= 255

val_X /= 255



print("Training dataset X shape:", train_X.shape)

print("Training dataset labels shape:", train_Y.shape)

print("Validation dataset X shape:", val_X.shape)

print("Validation dataset labels shape:", val_Y.shape)

    
#show sample augmented images

print("Augmented sample images:")

aug_samp1 = aug.flow(train_X)[0][0]

aug_samp1 = aug_samp1.reshape((28, 28))



plt.imshow(aug_samp1, cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()



aug_samp2 = aug.flow(train_X)[1][0]

aug_samp2 = aug_samp2.reshape((28, 28))



plt.imshow(aug_samp2, cmap="gray", interpolation="nearest")

plt.axis("off")

plt.show()
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



#Create model:

model = Sequential()



#CONV BLOCK 1 ******************************************************



#layer 1 - 32 3x3 filters applied with a stride of 1 and 0 padding, using relu activation before outputs passed to next layer

#input shape = 28x28x1

#output shape = 26x26x32

#number of trainable parameters = (3*3*1+1)*32 = 320

in_shape = train_X[0].shape

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=in_shape))



#layer 2 - 64 3x3 filters applied with a stride of 1 and 0 padding, using relu activation before outputs passed to next layer

#input shape = 26x26x32

#output shape = 24x24x64

#number of trainable parameters = (3*3*32+1)*64 = 18496

model.add(Conv2D(64, (3, 3), activation='relu'))



#pooling layer 1 - a 2x2 filter to halve the input in both dimensions by taking the maximum value (MaxPooling) in each 

#2x2 window it passes over in order to reduce the size of the volume to control overfitting and # parameters

#input shape = 24x24x64

#output shape = 12x12x64 

#number of trainable parameters = 0

model.add(MaxPooling2D(pool_size=(2, 2)))



#dropout layer 1 - at this point a random 1/4 of the input values will be set to 0 temporarily during training - forcing the other

#weights at the next layer to train more. this prevents overfitting as well and is useful for our small dataset.

#number of trainable parameters = 0

model.add(Dropout(0.25))





#DENSE LAYERS ******************************************************

#flatten layer - flattens the 3D input into a 1D array

#input shape = 12x12x64

#output shape = 9216

#number of trainable parameters = 0

model.add(Flatten())



#Dense layer 1 - a regular densely connected layer with 128 nodes, each with 9216 weights (one for each input = densely connected)

#also using relu activation

#input shape = 9216

#output shape = 128

#number of trainable parameters = 1179776

model.add(Dense(128, activation='relu'))



#dropout layer 2 - another random 1/2 of the input values will be set to 0 temporarily during training - forcing the other

#weights at the final layer to train more

model.add(Dropout(0.25))



#Dense layer 2 - a regular densely connected layer with 10 nodes, each with 128 weights (one for each input = densely connected)

#this time using softmax activation to force the output to represent a probability for each of the 10 output indexes 

#each of which coresponds to a possible output class (A-J)

#input shape = 9216

#output shape = 10 -> one for each class

#number of trainable parameters = 1290

model.add(Dense(10, activation='softmax'))



#Total trainable parameters = 1,199,882



#Compile the model using categorical cross entropy (for multi-class classification problems) and Adam optimizer to modify gradient 

#descent to use per-parameter learning rates and RMSProp to adapt learning rates based on velocity 

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.summary()
from keras.callbacks import EarlyStopping



epochs = 1 #setting to 1 to commit because kaggle runs all the code when it commits and it would take forever with 200

batch_size = 128 



# fit the model to the augmented training set, running through the training data [epochs] times 

earlystop = EarlyStopping(monitor='loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')

H = model.fit_generator(

    aug.flow(train_X, train_Y, batch_size=batch_size),

    validation_data=[val_X, val_Y],

    epochs=epochs, verbose=1, steps_per_epoch=1000, callbacks=[earlystop])
#Evaluate performance, describe training process:

# The above fit function outputs the accuracy on the validation set at the end of each epoch.

# After 20 training iterations with our first configuration, the accuracy began to plateau at 94%.

# We tried using Adadelta optimizer instead of Adam and switched from a batch size of 32 to 128 and saw the accuracy increase

# to a plateau of around 94% at around 20 epochs, with a loss of 0.2. Keras' docs say Adadelta performs best when the default

# parameters are used, so we didn't change those. 



# We changed the second last Dense layer from 128 nodes to 32, but the accuracy plateaued at 93%, so we changed it back.

# We changed the dropout on the last layer from 0.5 to 0.25

# We then added early stopping so that we could set the number of epochs to 200 and run it overnight to see if training for longer

# would overcome the plateau. 

# It stopped at 82 iterations with an accuracy of 95% and a loss of 0.197, so this was the final configuration we used.