import numpy as np 

import math

import pandas as pd 

import cv2

import matplotlib.pyplot as plt

from PIL import Image



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications import ResNet50

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline



import time



import os

print(os.listdir("../input"))
train_labels = pd.read_csv('../input/train_labels.csv', dtype=str)

test_labels = pd.read_csv('../input/sample_submission.csv', dtype=str)



#print('train : ','\n', train_sample.head(5))

#print('test : ','\n', test_labels.head(5))
#This image is labelled as having a cancer cell.

image = plt.imread('../input/train/c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')

plt.imshow(image)

plt.show()
num_classes = 2

my_model = Sequential()

my_model.add(ResNet50(include_top=False, weights='imagenet'))

my_model.add(Dense(num_classes, activation = 'softmax'))

my_model.layers[0].trainable = False
my_model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#Ratio of images in train compared to test

sample_size = 2000

ratio = 0.9 

sample_train = train_labels[:sample_size]

sample_test = test_labels[:sample_size]



size_train = math.ceil(ratio*sample_train.shape[0])

train_df=sample_train[:size_train]

test_df=sample_test[size_train+1:]



print('sample size : ', sample_size, '\n',

      'ratio : ', ratio,'\n',

      'train size : ', train_df.shape[0],'\n',

      'test size : ', test_df.shape[0])
# used in the reference url: https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

def append_ext(fn):

    return fn + '.tif'
train_df['id']=train_df['id'].apply(append_ext)

test_df['id']=test_df['id'].apply(append_ext)
train_df.head()
train_batch_size = 10

val_batch_size = 10

valid_ratio = 0.25



train_steps = np.ceil(train_df.shape[0] / train_batch_size)

val_steps = np.ceil((train_df.shape[0]*valid_ratio) / val_batch_size)
data_generator = ImageDataGenerator(rescale = 1./255., validation_split=valid_ratio)



train_generator = data_generator.flow_from_dataframe(dataframe = train_df, 

                                                directory = '../input/train/',

                                               x_col = 'id',

                                               y_col = 'label',

                                               subset = 'training',

                                               batch_size = train_batch_size,

                                               shuffle = True,

                                               class_mode = 'categorical',

                                               target_size = (96, 96))

validation_generator = data_generator.flow_from_dataframe(dataframe = train_df,

                                                         directory = '../input/train/',

                                                        x_col = 'id',

                                                        y_col = 'label',

                                                        subset = 'validation',

                                                        batch_size=val_batch_size,

                                                        shuffle = True,

                                                        class_mode = 'categorical',

                                                        target_size = (96, 96))



test_datagen = ImageDataGenerator(rescale = 1./255.)



test_generator = test_datagen.flow_from_dataframe(dataframe = test_df,

                                                directory = '../input/test/',

                                               x_col = 'id',

                                               y_col = 'label',

                                               class_mode = None,

                                               shuffle = False,

                                               target_size = (96, 96))
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



print('step size for : ', '\n', 'train : ', STEP_SIZE_TRAIN,

     '\n', 'valid : ', STEP_SIZE_VALID,

     '\n', 'test : ', STEP_SIZE_TEST)



my_model.fit_generator(generator = train_generator,

                       steps_per_epoch=STEP_SIZE_TRAIN,

                       validation_data = validation_generator,

                       validation_steps=STEP_SIZE_VALID,

                       epochs = 3)
evaluation = my_model.evaluate(x= val_img_array, y=val_img_label)

print()

print ("Loss = " + str(evaluation[0]))

print ("Test Accuracy = " + str(evaluation[1]))
print('number of images labelled with cancer : ',test_sample[test_sample['label']==1].shape[0],

      ' out of ', test_sample.shape[0], ' examples')
test_sample.to_csv('test_predictions.csv', index=False)