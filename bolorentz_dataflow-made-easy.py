import os

import pandas as pd

print(os.listdir("../input/data"))

from pylab import rcParams

rcParams['figure.figsize'] = 13, 13
import matplotlib.pyplot as plt

import numpy as np

import random



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow import keras as K

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(

        rescale=1./255, # This rescales the pixel brightnesses from 0-255 to 0-1

        validation_split = 0.1) 



test_datagen = ImageDataGenerator(rescale=1./255)



# Now we set our training generator to read images from the training directory

train_generator = train_datagen.flow_from_directory(

        '../input/data/train', # Az elérési út

        batch_size=32, # Images are not read and sent to the network individually, but in batches

        class_mode='categorical', # We spacify that we wish to use each subfolder in the training directory a s a separate class

        subset="training")



# The validation generator is constructed from the same object, with the same arguments, only this time from the set apart 10 percent from the files 

valid_generator = train_datagen.flow_from_directory(

        '../input/data/train',

        batch_size=32,

        class_mode='categorical',

        subset="validation") # Here we request these files to come from the validation set



# The test generator will stream form the test directory, so formally there is only one class, "test"

test_generator = test_datagen.flow_from_directory(

        '../input/data/',

        shuffle=False, # Now we do not want to shuffle the examples. Not at all

        classes=['test'],

        batch_size=32)
im = train_generator.next()

print(im[0].shape) # 32 Images as a batch

print(im[1].shape) # The labels are already one-hot encoded for us!

plt.imshow(im[0][0])
a = "relu"

model = Sequential()

model.add(Conv2D(8, (3, 3), activation=a, input_shape = ( 256, 256, 3)))

model.add(MaxPool2D(2, 2))

model.add(Conv2D(16, (3, 3), activation=a))

model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3, 3), activation=a))

model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3, 3), activation=a))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation=a))

model.add(Dropout(0.4))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

model.summary()
history = model.fit_generator(train_generator, epochs = 10, validation_data=valid_generator)
plt.plot(history.epoch, history.history["acc"])

plt.plot(history.epoch, history.history["val_acc"])
predictions = model.predict_generator(test_generator) # And this is how you predict
print(predictions)

print()

print(np.argmax(predictions, axis=1)) # Argmax grabs the most probable sport for each image
sample = pd.read_csv("../input/data/sample_submission.csv")

print(sample.head())

print(valid_generator.class_indices) #The class names are returned as a name-index mapping. We have to invert it

inv_map = {v: k for k, v in valid_generator.class_indices.items()}

print(inv_map)

submission = pd.DataFrame(data= {'ID': test_generator.filenames , 'sport' : np.argmax(predictions, axis=1)} )

submission = submission.replace(inv_map)

print(submission.head())

filename = 'Sports.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)