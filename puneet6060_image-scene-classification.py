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
import sys

import os

from keras.layers import *

from keras.optimizers import *

from keras.applications import *

from keras.models import Model, Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as k
# fix seed for reproducible results (only works on CPU, not GPU)

seed = 9

np.random.seed(seed=seed)

tf.set_random_seed(seed=seed)



# hyper parameters for model

nb_classes = 6  # number of classes

based_model_last_block_layer_number = 86  # value is based on based model selected.

img_width, img_height = 150, 150  # change based on the shape/structure of your images

batch_size = 128  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

nb_epoch = 50  # number of iteration the algorithm gets trained.

learn_rate = 1e-4  # sgd learning rate

momentum = .9  # sgd momentum to avoid local minimum

transformation_ratio = .2  # how aggressive will be the data augmentation/transformation
data_dir = '../working'

train_data_dir = os.path.abspath('../input/seg_train/seg_train')  # Inside, each class should have it's own folder

validation_data_dir = os.path.abspath('../input/seg_test/seg_test')  # each class should have it's own folder

model_path = '../working'

# Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.

# To save augmentations un-comment save lines and add to your flow parameters.

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                    shear_range=transformation_ratio,

                                    zoom_range=transformation_ratio,

                                   rotation_range=20,

                                    width_shift_range=transformation_ratio,

                                    height_shift_range=transformation_ratio,

                                   cval=transformation_ratio,

                                   horizontal_flip=True,

                                   vertical_flip=True)



validation_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(train_data_dir,

                                                    target_size=(img_width, img_height),

                                                    batch_size=batch_size,

                                                    class_mode='categorical')

labels = (train_generator.class_indices)

print(labels)



# save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')

# save_prefix='aug',

# save_format='jpeg')

# use the above 3 commented lines if you want to save and look at how the data augmentations look like



validation_generator = validation_datagen.flow_from_directory(validation_data_dir,

                                                              target_size=(img_width, img_height),

                                                              batch_size=batch_size,

                                                              class_mode='categorical')
# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(nb_classes, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])



top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')

callbacks_list = [

    ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),

    EarlyStopping(monitor='val_acc', patience=5, verbose=0)

]



# Train Simple CNN

history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples // batch_size,

                    epochs=20,

                    validation_data=validation_generator,

                    validation_steps=validation_generator.samples // batch_size,

                    callbacks=callbacks_list)



import matplotlib.pyplot as plt



print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# at this point, the top layers are well trained and we can start fine-tuning

# convolutional layers from inception V3. We will freeze the bottom N layers

# and train the remaining top layers.



# let's visualize layer names and layer indices to see how many layers

# we should freeze:

# for i, layer in enumerate(base_model.layers):

#    print(i, layer.name)



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:

for layer in model.layers[:249]:

   layer.trainable = False

for layer in model.layers[249:]:

   layer.trainable = True



# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])



top_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')

callbacks_list = [

    ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),

    EarlyStopping(monitor='val_acc', patience=10, verbose=0)

]



# Train Simple CNN

history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples // batch_size,

                    epochs=50,

                    validation_data=validation_generator,

                    validation_steps=validation_generator.samples // batch_size,

                    callbacks=callbacks_list)







print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# save model

model_json = model.to_json()

with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:

    json_file.write(model_json)
# release memory

k.clear_session()
from keras.models import model_from_json



# load json and create model

json_file = open('../working/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("../working/top_model_weights.h5")

print("Loaded model from disk")
import numpy as np # linear algebra

from tqdm import tqdm

import os

import cv2



#Validating model on real senarios

#print(testFiles)

testImg = []





# Obtain images and resizing, obtain labels

fileName=[]

i=0

for img in tqdm(os.listdir('../input/seg_pred/seg_pred/')):

  fileName.append(img)

  testImg.append(cv2.resize(cv2.imread('../input/seg_pred/seg_pred/'+img), (150, 150)))

  i=i+1



testImg = np.asarray(testImg)  # Train images set

testImg = testImg / 255

#print(testImg)
print(testImg.shape)

testY = loaded_model.predict(testImg)

#print(testY)

predY = np.argmax(testY, axis = 1)

d = []

i=0

for pred in predY:

    d.append({'image_name': fileName[i], 'label': pred})

    i=i+1

output = pd.DataFrame(d)

output.to_csv('submission.csv',index=False)