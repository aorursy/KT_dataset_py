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
# Attempt at creating a CNN to classify patients as having autism or not 



# Used the following resources 

    # https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/

    # https://www.geeksforgeeks.org/python-image-classification-using-keras/

    # https://github.com/CShorten/KaggleDogBreedChallenge/blob/master/DogBreed_BinaryClassification.ipynb

    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    

# Load images

    # https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
# Set random seed for purposes of reproducibility

seed = 123
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

import os 
# dimensions of our images.

img_width, img_height = 150, 150



train_data_dir = '../input/autistic-children-data-set-traintestvalidate/train'

validation_data_dir = '../input/autistic-children-data-set-traintestvalidate/valid'

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])

nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

epochs = 100

batch_size = 100
print('no. of trained samples = ', nb_train_samples, ' no. of validation samples = ',nb_validation_samples)
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)
loss, accuracy = model.evaluate_generator(validation_generator)

print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
###Test: accuracy = 0.810000  ;  loss = 0.537789      100
print(model.summary())
### To use the model in the future - run the necessary steps above and load the model from the .h5 file 



# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# save model and architecture to single file

model.save("model_v2.h5")

print("Saved model to disk")
# load and evaluate a saved model

from keras.models import load_model

from keras.preprocessing.image import img_to_array, load_img

import numpy as np
# load model

model = load_model('model_v2.h5')

# summarize model.

model.summary()
print(model.summary())
# dimensions of our images.

img_width, img_height = 150, 150



# Introduce the picture you want to try below.

img = load_img('../input/tahminad/001.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 001:", "Preds:", preds, "Probs", probs)







img = load_img('../input/tahminad/002.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 002:", "Preds:", preds, "Probs", probs)







img = load_img('../input/tahminad/003.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 003:", "Preds:", preds, "Probs", probs)





img = load_img('../input/tahminad/004.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 004:", "Preds:", preds, "Probs", probs)





img = load_img('../input/tahminad/005.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 005:", "Preds:", preds, "Probs", probs)





img = load_img('../input/tahminad/006.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 006:", "Preds:", preds, "Probs", probs)



img = load_img('../input/tahminad/007.png',False,target_size=(img_width,img_height))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)

probs = model.predict_proba(x)

print("image 007:", "Preds:", preds, "Probs", probs)


