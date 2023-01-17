import numpy as np

import pandas as pd

from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.models import Sequential

from glob import glob

import matplotlib.pyplot as plt



from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import cv2

import glob

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from tensorflow.python.keras import backend as K

import plotly.graph_objects as go

import plotly.offline as py

autosize =False



from plotly.subplots import make_subplots

import plotly.graph_objects as go



%matplotlib inline
import pandas as pd
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

x1 = "96"

x2 = "96"

x_train_dim = "x_train_" + x1

x_test_dim = "x_test_" + x1
x_train = np.load('../input/siimisic-melanoma-resized-images/' + x_train_dim + ".npy")

x_train.shape
y = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv').target

from sklearn.utils import class_weight

 

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y),

                                                 y)
class_weights = dict(enumerate(class_weights))
from sklearn.model_selection import train_test_split



train_imgs, validation_imgs, y_train, y_val = train_test_split(x_train,y, test_size=0.2, random_state=1234)
import gc

del x_train

gc.collect()
# define parameters for model training

batch_size = 128

num_classes = 2

epochs = 50

input_shape = (96,96,3)
# we will use Adam optimizer

opt = Adam(lr=1e-5)



nb_train_steps = train_imgs.shape[0]//batch_size

nb_val_steps=validation_imgs.shape[0]//batch_size



print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
# Pixel Normalization and Image Augmentation

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

                                   horizontal_flip=True, fill_mode='nearest')



# no need to create augmentation images for validation data, only rescaling the pixels

val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(train_imgs, y_train, batch_size=32)

val_generator = val_datagen.flow(validation_imgs, y_val, batch_size=32)
del train_imgs,validation_imgs

gc.collect()
from keras.applications import vgg16

from keras.models import Model

import keras



vgg = vgg16.VGG16(include_top=False, weights='imagenet', 

                                     input_shape=input_shape)



output = vgg.layers[-1].output

output = keras.layers.Flatten()(output)

vgg_model = Model(vgg.input, output)



vgg_model.trainable = False

for layer in vgg_model.layers:

    layer.trainable = False

    

import pandas as pd

pd.set_option('max_colwidth', -1)

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
vgg_model.trainable = True



set_trainable = False

for layer in vgg_model.layers:

    if layer.name in ['block5_conv1', 'block4_conv1']:

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False

        

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Sequential

from keras import optimizers



model = Sequential()

model.add(vgg_model)

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))





model.compile(loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()],optimizer=opt)
model.fit_generator(train_generator, steps_per_epoch=nb_train_steps, epochs=epochs,

                              validation_data=val_generator, validation_steps=nb_val_steps,class_weight=class_weights, 

                              verbose=1)
del train_generator,val_generator

gc.collect()
x_test = np.load("../input/siimisic-melanoma-resized-images/" + x_test_dim + ".npy")
from numpy import expand_dims

def tta_prediction(datagen, model, image, n_examples):

    # convert image into dataset

    samples = expand_dims(image, 0)

    # prepare iterator

    it = datagen.flow(samples, batch_size=n_examples)

    # make predictions for each augmented image

    probs = model.predict_generator(it, steps=n_examples, verbose=0)

    #print(len(probs))    

    prob = np.mean(probs, axis=1)    

    return prob
from tqdm import tqdm

# configure image data augmentation

test_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

                                   horizontal_flip=True, fill_mode='nearest')

target=[]

#i = 0

for img in tqdm(x_test):

    prediction=tta_prediction(test_datagen,model,img,32)    

    target.append(prediction[0])
# submission file

sub=pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")

sub['target']=target

sub.to_csv('submission.csv', index=False)

sub.head()
for i in tqdm(range(int(9e6))): 

    pass