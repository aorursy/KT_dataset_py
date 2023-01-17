# checking whether we have loaded the datasets we need and their input names

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D, MaxPool2D

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import os

%matplotlib inline 

#sets the backend of matplotlib to the "inline" backend. With this backend, the output of plotting commands is displayed inline within frontends like in Jupyter notebook, directly below the code cell that produced it. 
data = "../input/covid-19-x-ray-10000-images/dataset"
os.listdir(data)



#os.listdir: returns a list containing the names of the entries in the directory given by the path
import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



# glob.glob: returns a possibly empty list of path names that match pathnamme, which must be a string containing 

# a pth specification. Pathname can be either absolute like /usr/src/Python-1.5/Makefile or relative like

# ../../Tools/*/*.gif and can contain shell-style wildcards.
normal_images = [] # creating an empty list to be later employed

for img_path in glob.glob(data + '/normal/*'): # tells the function to add normal images within our dataset into our created normal lung CT list

    normal_images.append(mpimg.imread(img_path)) # and additionally reads our image





fig = plt.figure()

fig.suptitle('normal lung')

plt.imshow(normal_images[0], cmap = 'gray')



covid_images = []

for img_path in glob.glob(data + '/covid/*'):

    covid_images.append(mpimg.imread(img_path))

    

fig = plt.figure()

fig.suptitle('covid infected lung')

plt.imshow(covid_images[0], cmap = 'gray')
print(len(normal_images))

print(len(covid_images))
IMG_W = 150

IMG_H = 150

CHANNELS = 3



INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

NB_CLASSES = 2

EPOCHS = 48

BATCH_SIZE = 6
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = INPUT_SHAPE)) #initializing the weights, input and output

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(Conv2D(250, (3, 3)))

model.add(Activation("relu"))



model.add(Conv2D(128, (3, 3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2, 2))

model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2, 2))



model.add(Conv2D(256, (2, 2)))

model.add(Activation("relu"))

model.add(MaxPool2D(2, 2))



model.add(Flatten())

model.add(Dense(32))

model.add(Dropout(0.25))

model.add(Dense(1))

model.add(Activation("sigmoid"))
model.compile(loss = "binary_crossentropy",

             optimizer = "rmsprop", 

             metrics = ['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(rescale = 1./225,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True,

                                  validation_split = 0.3)



train_generator = train_datagen.flow_from_directory(

    data, 

    target_size = (IMG_H, IMG_W),

    batch_size = BATCH_SIZE,

    class_mode = "binary",

    subset = "training")



# .flow_from_directory: to read the images from a big numpy array and folders containing images.



validation_generator = train_datagen.flow_from_directory(

    data,

    target_size = (IMG_H, IMG_W),

    batch_size = BATCH_SIZE,

    class_mode = "binary",

    shuffle = False,

    subset = "validation")



history = model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // BATCH_SIZE,

    validation_data = validation_generator,

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs = EPOCHS)



#.fit_generator: to perform data augmentation to avoid the overfitting of a model and also to increase the ability of our model to generalize.

# additional_info: https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("model accuracy")

plt.ylabel("accuracy")

plt.xlabel("epoch")

plt.legend(['train', 'test'], loc = "upper left")

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("model loss")

plt.ylabel("loss")

plt.xlabel("epoch")

plt.legend(['train', 'test'], loc = "upper left")

plt.show()
print("training_accuracy", history.history['accuracy'][-1])

print("validation_accuracy", history.history['val_accuracy'][-1])
label = validation_generator.classes
pred = model.predict(validation_generator)

predicted_class_indices = np.argmax(pred, axis = 1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k, v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]

print(predicted_class_indices)

print(labels)

print(predictions)
from sklearn.metrics import confusion_matrix



cf = confusion_matrix(predicted_class_indices, label)

cf
exp_series = pd.Series(label)

pred_series = pd.Series(predicted_class_indices)

pd.crosstab(exp_series, pred_series, rownames = ['Actual'], colnames = ['Predicted'], margins = True)
plt.matshow(cf)

plt.title("Confusion Matrix Plot")

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show();