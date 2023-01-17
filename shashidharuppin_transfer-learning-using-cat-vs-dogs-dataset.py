# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/model-evaluation-utils/model_evaluation_utils.py", dst = "../working/model_evaluation_utils.py")



# import all our functions

from model_evaluation_utils import *
#Importing all the dependencies required for the model

from keras.models import Model

import keras

from keras.applications import vgg16

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Sequential

from keras import optimizers

from matplotlib import pyplot

from matplotlib.image import imread

from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
#Create a data generator with some image augmentation.

input_path = '../input/dogs-vs-cat/'



train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_directory(

    input_path + 'train',

    batch_size=64,

    class_mode='binary',

    target_size=(224,224),

    color_mode="rgb")



validation_datagen = ImageDataGenerator(rescale=1./255,

    preprocessing_function=preprocess_input)



validation_generator = validation_datagen.flow_from_directory(

    input_path + 'valid',

    shuffle=False,

    class_mode='binary',

    target_size=(224,224),

    color_mode="rgb")



test_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input)



test_generator = test_datagen.flow_from_directory(

    input_path + 'test1',

    batch_size=64,

    class_mode='binary',

    target_size=(224,224),

    color_mode="rgb")
#Lets Plot some of the images of dogs and cat

# plot dog photos from the dogs vs cats valid dataset

# define location of dataset

folder = '../input/dogs-vs-cat/valid/dogs'

# plot first few images

for i in range(9):

    # define subplot

    pyplot.subplot(330 + 1 + i)

    # define filename

    filename = folder + '/dog.' + str(i) + '.jpg'

    # load image pixels

    image = imread(filename)

    # plot raw pixel data

    pyplot.imshow(image)

# show the figure

pyplot.show()
# plot cat photos from the dogs vs cats valid dataset

# define location of dataset

folder = '../input/dogs-vs-cat/valid/cat'

# plot first few images

for i in range(9):

    # define subplot

    pyplot.subplot(330 + 1 + i)

    # define filename

    filename = folder + '/cat.' + str(i) + '.jpg'

    # load image pixels

    image = imread(filename)

    # plot raw pixel data

    pyplot.imshow(image)

# show the figure

pyplot.show()


vgg = vgg16.VGG16(include_top=False, weights='imagenet', 

                                     input_shape=(224,224,3))



output = vgg.layers[-1].output

output = keras.layers.Flatten()(output)



vgg_model = Model(vgg.input, output)

vgg_model.trainable = False



for layer in vgg_model.layers:

    layer.trainable = False



vgg_model.summary()



import pandas as pd

pd.set_option('max_colwidth', -1)

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
input_shape = vgg_model.output_shape[1]



model = Sequential()

model.add(vgg_model)

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['accuracy'])



model.summary()
history = model.fit_generator(train_generator, 

                              steps_per_epoch=100, 

                              epochs=25,

                              validation_data=validation_generator, 

                              validation_steps=50, 

                              verbose=1

                             )
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, 25, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, 25, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
model.save('/kaggle/working/transer_learning.h5')
model.save_weights('/kaggle/working/transfer_learning_weights.h5')
_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=0)

print('> %.3f' % (acc * 100.0))
from keras.models import load_model

import model_evaluation_utils as meu



tl_img_aug_cnn = load_model('../working/transer_learning.h5')



test_datagen_new = ImageDataGenerator(

    rescale=1./255,

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input)



test_generator_new = test_datagen_new.flow_from_directory(

    input_path + 'test1',

    class_mode='binary',

    batch_size = None,

    target_size=(224,224),

    color_mode="rgb")



predictions = tl_img_aug_cnn.predict_classes(test_generator_new, verbose=0)

predictions = num2class_label_transformer(predictions)

model_evaluation_utils.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 

                                      classes=list(set(test_labels)))