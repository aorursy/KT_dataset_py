# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('./'))

# Any results you write to the current directory are saved as output.
from IPython.display import Image, display
import PIL
img_da_importunare = "../input/test_set/test_set/cats/cat.4002.jpg"

def show_img(percorso):
    if isinstance(percorso, str):
        display(Image(percorso))
    if isinstance(percorso, Image):
        display(percorso)
    if isinstance(percorso, PIL.Image.Image):
        display(percorso)
show_img(img_da_importunare)

#creo qualche oggetto per aiutarmi con l' "aumentazione" delle immagini
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
K.set_image_dim_ordering('th') #potrebbe essere da cavare

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1./255)
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../input/training_set/training_set',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
        '../input/test_set/test_set',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.to_json())

#preparo i checkpoint per i pesi, 
#dato che potrebbe smettere di funzionare tutto da un momento all altro

from keras.callbacks import ModelCheckpoint

weight_checkpoint_path="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(
    weight_checkpoint_path, 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True,
    mode='max')
    
model.fit_generator(
    train_generator,
    steps_per_epoch = 2000 // batch_size,
    epochs=50,
    validation_data = test_generator,
    validation_steps=800 // batch_size,
    #callbacks = [checkpoint]
)
model.save_weights('first_model_final_weight.h5')  # always save your weights after training or during training

