# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/natural-images/data/natural_images"))



# Any results you write to the current directory are saved as output.

######## Gpu Check ###########

import tensorflow as tf 

device_name = tf.test.gpu_device_name() 

if device_name != '/device:GPU:0':  

    raise SystemError('GPU device not found') 

print('Found GPU at: {}'.format(device_name)) 
# from google.colab import drive

# drive.mount('/content/drive/')
# import zipfile

# zip_ref = zipfile.ZipFile('../input/data/natural_images', 'r')

# zip_ref.extractall()

# zip_ref.close()
from os import listdir

img_dir = '../input/natural-images/data/natural_images'

data_list = listdir(img_dir)

data_list
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model ,load_model

from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import tensorflow as tf





DATASET_PATH  = img_dir

IMAGE_SIZE    = (299, 299)

NUM_CLASSES   = len(data_list)

BATCH_SIZE    = 25  

FREEZE_LAYERS = 16  

NUM_EPOCHS    = 1

LEARNING_RATE = 5e-5

DROP_OUT = .5

chkpoint_model_loc = '5'





train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   rotation_range=50,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.25,

                                   zoom_range=0.1,

                                   channel_shift_range = 20,

                                   horizontal_flip = True ,

                                   vertical_flip = True ,

                                   validation_split = 0.2,

                                   fill_mode='constant')



# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

#                                    fill_mode='constant')



train_batches = train_datagen.flow_from_directory(DATASET_PATH,

                                                  target_size=IMAGE_SIZE,

                                                  shuffle=True,

                                                  batch_size=BATCH_SIZE,

                                                  subset = "training"

                                                  )



valid_batches = train_datagen.flow_from_directory(DATASET_PATH,

                                                  target_size=IMAGE_SIZE,

                                                  shuffle=True,

                                                  batch_size=BATCH_SIZE,

                                                  subset = "validation"

                                                  )

class_dictionary = train_batches.class_indices

class_dictionary

len(chkpoint_model_loc)
# build our classifier model based on pre-trained InceptionResNetV2:

# 1. we don't include the top (fully connected) layers of InceptionResNetV2

# 2. we add a DropOut layer followed by a Dense (fully connected)

#    layer which generates softmax class score for each class

# 3. we compile the final model using an Adam optimizer, with a

#    low learning rate (since we are 'fine-tuning')

from tensorflow.keras.layers import GlobalAveragePooling2D

if len(chkpoint_model_loc) < 5:

#############main portion for InceptionResnetV2 only##############

  net = InceptionResNetV2(include_top=False,

                          weights='imagenet',

                          input_tensor=None,

                          input_shape=(299,299,3))



  ############## Run this block if include top = False ######

  x = net.output

  x = GlobalAveragePooling2D()(x)

#   x = Flatten()(x)

#   x = Dropout(DROP_OUT)(x)

###############   End of block ##################



############## Run this block if include top = True ######

#   '''

#   This block keeps avg_pool (GlobalAveragePooling2) layer and removes only prediction layer.

#   Then replace inceptionResnetv2's predictionlayer with a drpout layer and and a softmax layer. 

#   To change the class class size of 1000 to 8 we need to replace their pediction layer with our softmax layer.

#   '''

#   avg_pool = net.layers[-2]

#   prediction = net.layers[-1]



#   #create drpout layer

#   drp1 = Dropout(DROP_OUT)

#   x = drp1(avg_pool.output)

###############   End of block ##################







  output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

  net_final = Model(inputs=net.input, outputs=output_layer)



  for layer in net_final.layers[:FREEZE_LAYERS]:

      layer.trainable = False



  for layer in net_final.layers[FREEZE_LAYERS:]:

      layer.trainable = True



  net_final.compile(optimizer=Adam(lr=LEARNING_RATE),

                    loss='categorical_crossentropy', metrics=['accuracy'])



#   print(net_final.summary())
print(net_final.summary())
#CHECKPOINT

# filepath = "../weights-improvement-EPOCH_{epoch:02d}-ACC_{acc:.2f}-VALIDATION_{val_acc:.2f}.hdf5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]
######## if running pre trained model is running #######

# if len(chkpoint_model_loc) > 5:

#   net_final = load_model(chkpoint_model_loc)

#   net_final.evaluate_generator(valid_batches,

#                                steps = np.ceil(len(valid_batches) / BATCH_SIZE),

#                                verbose = 1

#                                )

if len(chkpoint_model_loc) > 5:

  net_final.evaluate_generator(valid_batches,

                               steps = np.ceil(len(valid_batches) / BATCH_SIZE),

                               verbose = 1

                               )

#Adding Dropout layer to a pre trained model

drpout_needed = False



if drpout_needed :

  flt = net_final.layers[-2]

  prediction = net_final.layers[-1]



  #create drpout layer

  drp1 = Dropout(DROP_OUT)

  x = drp1(flt.output)



  predictors = prediction(x)

  net_final = Model(inputs=net_final.input, outputs=predictors)
print(net_final.summary())
# !pip install tensorflow-gpu==2.0.0-alpha0

# import tensorflow as tf

# print(tf.__version__)


#FIT MODEL

net_final.fit_generator(train_batches,

                        steps_per_epoch = np.ceil(len(train_batches) / BATCH_SIZE),

                        validation_data = valid_batches,

                        validation_steps = np.ceil(len(valid_batches) / BATCH_SIZE),

                        epochs = NUM_EPOCHS,

#                         callbacks = callbacks_list 

                       )

# # save trained weights

# # net_final.save(WEIGHTS_FINAL)

# x = net_final.evaluate_generator(valid_batches,

#                            steps = np.ceil(len(valid_batches) / BATCH_SIZE),

#                            use_multiprocessing = True,

#                            verbose = 1

#                            )
# incorrects = np.nonzero(net_final.predict(valid_batches).reshape((-1,)) != valid_batches)
# from PIL import Image

import pandas as pd

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt



IMAGE_SIZE    = (299, 299)

# print(data_list)

print("Class name to class id map\n",class_dictionary)



test_image = image.load_img((img_dir + "/motorbike/motorbike_0011.jpg"),target_size =IMAGE_SIZE )

test_image = image.load_img("../input/natural-images/data/natural_images/cat/cat_0005.jpg",target_size =IMAGE_SIZE )

test_image = image.img_to_array(test_image)



plt.imshow(test_image/255.)



# test_image = np.expand_dims(test_image , axis = 0)

test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))

test_image = preprocess_input(test_image)



prediction = net_final.predict(test_image)

df = pd.DataFrame({'pred':prediction[0]})

# print(prediction[0])

df = df.sort_values(by='pred', ascending=False, na_position='first')

print(df)



for x in data_list:

  if class_dictionary[x] == (df[df == df.iloc[0]].index[0]):

    print("Class prediction = ", x)

    break
os.listdir('../input/natural-images/data/natural_images')


# import sys

# import numpy as np

# import cv2

# import utils

# import matplotlib.pyplot as plt





# pred = net_final.predict(test_image)



# bboxes = utils.get_boxes(pred[0], cutoff=0.1)

# bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.05)

# draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, draw_dot=True, radius=3)

# draw = draw.astype(np.uint8)



# plt.imshow(draw[...,::-1])

# plt.show()
# # from PIL import Image

# import pandas as pd

# from tensorflow.keras.preprocessing import image

# import matplotlib.pyplot as plt



# IMAGE_SIZE    = (299, 299)



# test_image = image.load_img("../input/data/natural_images/motorbike/motorbike_0000.jpg",target_size =IMAGE_SIZE )

# test_image = image.img_to_array(test_image)

# plt.imshow(test_image/255.)

# test_image = np.expand_dims(test_image , axis = 0)



# prediction = net_final.predict(test_image)

# df = pd.DataFrame({'pred':prediction[0]})

# print(prediction[0])

# df.sort_values(by='pred', ascending=False, na_position='first')