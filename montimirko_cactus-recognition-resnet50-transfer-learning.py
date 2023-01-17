# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.preprocessing.image as image

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
train_dir = '../input/train/train'

test_dir = '../input/test/test'



train = pd.read_csv('../input/train.csv')



print('train has {} rows'.format(len(os.listdir(train_dir))))

print('test has {} rows'.format(len(os.listdir(test_dir))))

train['has_cactus'] = train['has_cactus'].astype(str)



print(train['has_cactus'].dtype)

#train['has_cactus'] = train['has_cactus'].apply(lambda x : int(x))

train_datagen = image.ImageDataGenerator(1.0/255,

                                        preprocessing_function=preprocess_input,

                                        horizontal_flip=True,

                                        width_shift_range = 0.2,

                                        height_shift_range = 0.2)



valid_datagen = image.ImageDataGenerator(1.0/255,

                                        preprocessing_function=preprocess_input

                                        )



                                   

            

            

train_generator = train_datagen.flow_from_dataframe(dataframe=train[:15001],

                                                   directory=train_dir,

                                                   x_col='id',

                                                   y_col='has_cactus',

                                                   class_mode = 'binary',

                                                   batch_size=64,

                                                   target_size=(150,150))
valid_generator = valid_datagen.flow_from_dataframe(dataframe=train[15000:],

                                                   directory=train_dir,

                                                   x_col='id',

                                                   y_col='has_cactus',

                                                   class_mode='binary',

                                                   batch_size=64,

                                                   target_size=(150,150))
resnet_weights_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



filepath='weights.best.hdf5'

es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=5,baseline=0.99)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')





def myModel():

    my_model2 = keras.models.Sequential()

    input_layer = keras.layers.Input(shape=(150, 150, 3), name='image_input')

    my_model2.add(ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer))

    my_model2.add(keras.layers.Flatten())

    my_model2.add(keras.layers.Dense(16))

    my_model2.add(keras.layers.BatchNormalization())

    my_model2.add(keras.layers.Activation('relu'))

    my_model2.add(keras.layers.Dropout(0.8))  #add strong normalization against overfitting

    my_model2.add(keras.layers.Dense(1, activation='sigmoid'))

    my_model2.layers[0].trainable=True

    my_model2.summary()

    return my_model2



my_model2 = myModel()
my_model2.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



history_2 = my_model2.fit_generator(train_generator,

                                   steps_per_epoch=50,

                                   epochs=15,

                                   validation_data= valid_generator,

                                   validation_steps=50,

                                   callbacks=[es, checkpoint])
my_model2 = myModel()

my_model2.load_weights(filepath)



my_model2.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



print("Created model and loaded weights from file")

scores = my_model2.evaluate_generator(valid_generator)

print("Accuracy = ", scores[1])
import matplotlib.pyplot as plt



acc = history_2.history['acc']

epochs_2 = range(0,15)



plt.plot( epochs_2,acc,label='training accuracy' )



plt.xlabel('n epochs')

plt.ylabel('accuracy')



acc_val = history_2.history['val_acc']

plt.plot(epochs_2, acc_val,label="validation accuracy")

plt.title('epochs vs acc')

plt.legend()





import cv2

from scipy import ndimage, misc



img = cv2.imread('../input/train/train/008bd3d84a1145e154409c124de7cee9.jpg', flags=cv2.IMREAD_COLOR)

plt.imshow(img)

plt.show()





filepath  = '../input/train/train/028192187883168e2a7621c998dc447a.jpg'

image = ndimage.imread(filepath, mode="RGB")

image_resized = misc.imresize(image, (150, 150,3))



plt.imshow(image_resized)

plt.show()





image_resized = np.reshape(image_resized,[1,150,150,3])



print(my_model2.predict(image_resized))




