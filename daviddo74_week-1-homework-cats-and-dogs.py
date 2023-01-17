# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
N_CLASSES = 2

BATCH_SIZE = 64

W = H = 128

classes = ['cat', 'dog']
train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0, 

                                                               brightness_range=(0.5, 2),

                                                               height_shift_range = 0.25,

                                                               width_shift_range = 0.25,

                                                               zoom_range = 0.5,

                                                               shear_range = 0.5,

                                                               horizontal_flip=True)
test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
train_dataset = train_generator.flow_from_directory(directory='../input/catndog/catndog/train/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')



test_dataset = test_generator.flow_from_directory(directory='../input/catndog/catndog/test/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')
grids = (5,5)

counter = 0



plt.figure(figsize=(10,10))



for batch_images, batch_labels in train_dataset:

    i = np.random.randint(len(batch_images))

    i = 0

    img = batch_images[i]

    label = batch_labels[i]

    

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter)

    ax = plt.imshow(img, cmap='brg')

    plt.xticks([])

    plt.yticks([])

    plt.title(classes[int(label)])
grids = (5,5)

counter = 0



plt.figure(figsize=(10,10))



for batch_images, batch_labels in test_dataset:

    i = np.random.randint(len(batch_images))

    i = 0

    img = batch_images[i]

    label = batch_labels[i]

    

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter)

    ax = plt.imshow(img, cmap='brg')

    plt.xticks([])

    plt.yticks([])

    plt.title(classes[int(label)])
vgg = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(H, W, 3))

len_vgg_layers = len(vgg.layers)

print("Total number of vgg layers: ", len_vgg_layers)



vgg_outputshape = vgg.output_shape

print("Vgg output shape from input shape (%d, %d, 3): "%(H, W), ":", vgg_outputshape)
# define top model

flatten = keras.layers.Flatten()(vgg.output)

fc = keras.layers.Dense(256, activation='relu')(flatten)

prob = keras.layers.Dense(1, activation='sigmoid')(fc)
model = keras.models.Model(vgg.input, prob)

model
for i in range(len_vgg_layers):

    model.layers[i].trainable = False
model.summary()

model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
model.fit_generator(train_dataset, epochs=10, steps_per_epoch=30, validation_data=test_dataset, validation_steps=10)
test_sample_images, test_sample_labels = next(test_dataset)

# make prediction

predict_sample_labels = (model.predict_on_batch(test_sample_images) > 0.5).astype(int)
grids = (3,3)

counter = 0



plt.figure(figsize=(10,10))



for img, gt_label, predict_label in zip(test_sample_images, test_sample_labels, predict_sample_labels):

    

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter)

    ax = plt.imshow(img, cmap='brg')

    plt.xticks([])

    plt.yticks([])

    plt.title("Actual: %s    Predict: %s"%(classes[int(gt_label)], classes[int(predict_label)]))
#plt.imshow(image[0])
#aug_iter = datagen.flow(image)

#aug_images = [next(aug_iter)[0].astype(np.int) for i in range(10)]
#aug_images = [next(aug_iter)[0].astype(np.int) for i in range(10)]
#aug_images[0].shape
#for i in range(0, 9):

 # ax = plt.subplot(3, 3, i+1)

  #ax = plt.imshow(aug_images[i])
