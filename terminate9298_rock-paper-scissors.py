%matplotlib inline

import pandas as pd

import os,shutil,math,scipy,cv2

import numpy as np

import matplotlib.pyplot as plt



import random as rn





from sklearn.utils import shuffle

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,roc_curve,auc



from PIL import Image

from PIL import Image as pil_image

from PIL import ImageDraw



from time import time

from glob import glob

from tqdm import tqdm

from skimage.io import imread

from IPython.display import SVG



# from scipy import misc,ndimage

# from scipy.ndimage.interpolation import zoom

# from scipy.ndimage import imread

print(os.listdir("../input"))



from keras import backend as K

from keras.models import load_model

from keras.utils.np_utils import to_categorical

from keras import layers

from keras.preprocessing.image import save_img

from keras.utils.vis_utils import model_to_dot

from keras.applications.vgg16 import VGG16,preprocess_input

from keras.applications.mobilenet import MobileNet

from keras.models import Sequential,Input,Model

from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D

from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
%%time

data_dir = '../input/rps-cv-images/'

augs_gen = ImageDataGenerator(

    rescale=1./255,        

    horizontal_flip=True,

    height_shift_range=.2,

    vertical_flip = True,

    validation_split = 0.2

)  

train_gen = augs_gen.flow_from_directory(

    data_dir,

    target_size = (224,224),

    batch_size = 32,

    class_mode = 'categorical',

    shuffle = True

    )

val_gen = augs_gen.flow_from_directory(

    data_dir,

    target_size = (224,224),

    batch_size = 32,

    class_mode = 'categorical',

    shuffle = True,

    subset='validation'

    )

%%time

model = Sequential()

model.add(layers.Conv2D(32 , (3,3) , input_shape=(224,224,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(64 ,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size= (2,2)))



model.add(layers.Conv2D(64 , (3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size= (2,2)))



model.add(layers.Flatten())

model.add(layers.Dense(512))

model.add(layers.Activation('relu'))   

model.add(layers.Dropout(0.5))

model.add(layers.Dense(64))

model.add(layers.Activation('relu'))   

model.add(layers.Dropout(0.5))

model.add(layers.Dense(3,activation='softmax'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics =['accuracy'])

model.summary()
%%time

history = model.fit_generator(

    train_gen, 

    steps_per_epoch=50,

    epochs=10,

    validation_data=val_gen,

    validation_steps= 50

)
model.save_weights('model_wieghts_complete.h5')

model.save('model_keras_complete.h5')
%%time

model_score = model.evaluate_generator(val_gen,steps=20)

print("Model Test Loss:",model_score[0])

print("Model Test Accuracy:",model_score[1])

%%time

model_base = MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))

model_new = Sequential()

model_new.add(model_base)

model_new.add(GlobalAveragePooling2D())

model_new.add(Dropout(0,5))

model_new.add(Dense(3,activation='softmax'))

model_new.compile(

    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



model_new.summary()

%%time

history = model_new.fit_generator(

    train_gen, 

    steps_per_epoch  = 50, 

    validation_data  = val_gen,

    validation_steps = 50,

    epochs = 10, 

    verbose = 1

)
model_score = model_new.evaluate_generator(val_gen,steps=20)

print("Model Test Loss:",model_score[0])

print("Model Test Accuracy:",model_score[1])
model.save_weights('model_new_wieghts_complete.h5')

model.save('model_new_keras_complete.h5')
# %%time

# data_dir_test = '../input/rock-scissors-paper-test-dataset/'

# augs_gen_test = ImageDataGenerator(

#     rescale=1./255

# )  

# test_gen = augs_gen_test.flow_from_directory(

#     data_dir_test,

#     batch_size = 32,

#     target_size = (224,224),

#     class_mode = 'binary'

#     )

# test_result = model_new.predict_generator(test_gen,1)
# test_result