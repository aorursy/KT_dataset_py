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
import matplotlib.pyplot as plt 

import cv2 

import imutils

from os import listdir

import os

%matplotlib inline



import sys

# sys.executable

import pandas as pd

import keras as K

from  keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import VGG16

from keras.applications.resnet import ResNet50

from keras.applications.mobilenet import MobileNet

from keras.applications.vgg19 import VGG19
!pip install imutils
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K



# set the matplotlib backend so figures can be saved in the background

import matplotlib

# matplotlib.use("Agg")

 

# import the necessary packages

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

# from pyimagesearch.smallervggnet import SmallerVGGNet

import matplotlib.pyplot as plt

from imutils import paths

import numpy as np

import argparse

import random

import pickle

import cv2

import os

import pandas as pd



import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras.models import load_model


"""Use below line to run notebookk locally """

# train_img_path='../dataset/Train Images/'

# test_img_path='../dataset/Test Images/'

# Train_csv_path='../dataset/train.csv'

# Test_csv_path='../dataset/test.csv'



"""below line to run notebook on collab"""

train_img_path='/kaggle/input/dataset/Train Images/'

test_img_path='/kaggle/input/dataset/Test Images/'

Train_csv_path='/kaggle/input/dataset/original_and_flipped.csv'

Test_csv_path='/kaggle/input/dataset/test.csv'

model_path='./content/model10.model'

flipped_img_path='/kaggle/input/dataset/Train_flipped/'

output_path='/kaggle/working/'



image_dims=[224,224,3] 

"""suggested [60 80 3]"""

batch_size=32

epochs=10

lr=1e-2

args={}

args['model']=None

args['output']=output_path
pwd


data=pd.read_csv(Train_csv_path)

data.head(2)


dummy_cols=pd.Series(data['Class'].unique())

label_dict=pd.get_dummies(dummy_cols).astype(int)

label_dict


data.shape
class CNN:

    @staticmethod

    def build(width, height, depth, outputshape):

        # initialize the model along with the input shape to be

        # "channels last" and the channels dimension itself

        model = Sequential()

        # CONV => RELU => POOL



        inputShape = (height, width, depth)

        chanDim = -1



        # if we are using "channels first", update the input shape

        # and channels dimension

        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3)))

        # model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Dropout(0.25))

        		# first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation("relu"))

        model.add(BatchNormalization())



        model.add(Dense(512))

        model.add(Activation("relu"))

        model.add(BatchNormalization())



        model.add(Dense(256))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        # model.add(Dropout(0.5))



        # 

        model.add(Dense(outputshape))

        model.add(Activation("softmax"))



        # return the constructed network architecture

        return model
class VGG:

  @staticmethod

  def build(width, height, depth, output_shape):

    custom_vgg = Sequential()

    custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (width, height,depth)))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(MaxPooling2D((2, 2)))



    custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(MaxPooling2D((2, 2)))



    custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(MaxPooling2D((2, 2)))





    custom_vgg.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(MaxPooling2D((2, 2)))



    custom_vgg.add(Conv2D(512, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Conv2D(512, (3, 3), strides = 1, padding = "same", activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(MaxPooling2D((2, 2)))



    custom_vgg.add(Flatten())



    custom_vgg.add(Dense(1024, activation = "relu"))

    custom_vgg.add(Dropout(0.4))

    custom_vgg.add(Dense(1024, activation = "relu"))

    custom_vgg.add(Dropout(0.4))



    custom_vgg.add(Dense(512, activation = "relu"))

    custom_vgg.add(Dropout(0.2))

    custom_vgg.add(Dense(512, activation = "relu"))

    custom_vgg.add(Dropout(0.2))



    custom_vgg.add(Dense(output_shape, activation = "softmax"))



  # custom_vgg.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return custom_vgg

from keras.applications.vgg16 import preprocess_input

class generator():

  def __init__(self,mode='train', csv_path=Train_csv_path,flip_img_dir=flipped_img_path, img_dir=train_img_path, label_dict=label_dict):

    self.mode=mode

    self.image_paths=sorted(list(listdir(img_dir)))

    self.image_paths+=sorted(list(listdir(flip_img_dir)))

    self.img_dir=[img_dir,flip_img_dir]

    self.label_dict=label_dict

    self.data=pd.read_csv(csv_path)

    self.index=0

    self.csv_data=dict(zip(self.data['Image'].values,self.data['Class'].values))

    random.seed(10)

    random.shuffle(self.image_paths)

    if mode=='test':

      self.image_paths=self.image_paths[10000:]

    else:

      self.image_paths=self.image_paths[:10000]



  def generate_image(self,batch_size=batch_size):



      

    while True:

      # index*batch_size<len(image_paths)

      batch_x=[]

      batch_y=[]

      # batch_imgs=np.random.choice(image_paths,size=batch_size)

      batch_imgs= self.image_paths[(self.index)*batch_size:(self.index+1)*batch_size]

      for i in batch_imgs:

        try:

          temp_img=cv2.imread(self.img_dir[0]+i)

          temp_img.shape

          # print('in try ', self.img_dir[0]+i)

        except:

          # print('in catch ', self.img_dir[1]+i)

          temp_img=cv2.imread(self.img_dir[1]+i)

        temp_img.shape

        temp_img=cv2.resize(temp_img,(image_dims[0],image_dims[1]))

        # temp_img -= temp_img.mean()

        # temp_img /= np.maximum(temp_img.std(), 1/image_dims[0]**2)

        img = img_to_array(temp_img)

        # img = np.expand_dims(img, axis=0)

        img = preprocess_input(img)

        batch_x.append(img)

        batch_y.append(self.label_dict[self.csv_data[i]].values)

        # except:

        #   print('not found')

        #   pass

      x=np.array(batch_x, dtype='float')

      y=np.array(batch_y)

      self.index+=1

      if self.mode=='train' and (self.index)==(10000//batch_size):

        self.index=0

      elif self.mode=='test' and (self.index)==1966//batch_size:

        self.index=0

      if x.shape[0]!=32:

        print(x.shape,'  not complete batch ')

      yield (x,y)
from keras.callbacks import Callback

class EpochCheckpoint(Callback):

  def __init__(self,output_path,name,start_at=0,every=5):

    super(Callback,self).__init__()

    self.output_path=output_path

    self.every=every

    self.intEpoch=start_at

    self.name=name

  def on_epoch_end(self,epoch,logs={}):

    if (self.intEpoch+1)%self.every==0:

      self.model.save(self.output_path+f'{self.name}_{self.intEpoch+1}.hdf5',overwrite=True)

    self.intEpoch+=1



def get_custom_model(model_name='vgg',lr=lr):

  if model_name=='vgg':

    model=VGG.build(image_dims[1],image_dims[0],image_dims[2],output_shape=4)

  elif model_name=='cnn':

    model=CNN.build(image_dims[0],image_dims[1],image_dims[2],output_shape=4)    

  opt = Adam(lr=lr)

  model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

  return model
def get_model(lr=lr, model=VGG16):

# initialize the model

    print("[INFO] compiling model...")

    base_model=model(include_top=False,weights='imagenet')#,input_shape=(image_dims[1],image_dims[0],image_dims[2]))

    # x=base_model.layers[-1].output

    x=base_model.output

    # x=Flatten()(x)

    x=GlobalAveragePooling2D()(x)

    x=Dense(512,activation='relu')(x) #dense layer 1

    x=Dropout(0.5)(x)

    x=Dense(512,activation='relu')(x) #dense layer 2

    x=Dropout(0.5)(x)

    # x=Dense(512,activation='relu')(x) #dense layer 2

    # x=Dropout(0.5)(x)

    preds=Dense(4,activation='softmax')(x) #dense layer 3

    model=Model(input=base_model.input,outputs=preds)

    for i in model.layers[:-6]:

        i.trainable=False

    opt = Adam(lr=0.00005)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

    return model

def train(BS=batch_size,EPOCHS=epochs,lr=lr,model_name=ResNet50):# initialize the number of epochs to train for, initial learning rate,

    print("[INFO] training network...")

    if args['model']==None:

      # model=get_model()

      model=get_model(model=model_name)

    else :

      model=load_model(args['model'])

      opt=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

      for layer in model.layers:

        layer.trainable = True

      model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')



    print(f'old learning rate:  {K.get_value(model.optimizer.lr)}')

    K.set_value(model.optimizer.lr,lr)

    print(f' training on lr : {K.get_value(model.optimizer.lr)}')  



    """instance for training and testing generator"""



    H = model.fit_generator(generator=train_generator.generate_image(),

        validation_data=test_generator.generate_image(),

        validation_steps=61,

        steps_per_epoch=312,

        callbacks=callbacks,

        epochs=EPOCHS, verbose=1)

    return model,H
args['model']=None

callbacks=[EpochCheckpoint(output_path=args['output'],name='vgg16')]

test_generator=generator(mode='test')

train_generator=generator()

vgg16_10=train(lr=1e-3,EPOCHS=5, model_name=VGG16)
# model='vgg16_5.hdf5'

# # print(type(model))

# args['model']=model

# callbacks=[EpochCheckpoint(start_at=5,output_path=args['output'],name='vgg16')]

# vgg16_15=train(lr=0.5*1e-4,EPOCHS=10)
from keras.models import load_model

def prediction(img_path=test_img_path, model_path=args['model']):

  model=load_model(model_path)

  output=[]

  for img in listdir(img_path):

    temp_img=cv2.imread(img_path+img)

    # print(im.shape)

    temp_img=cv2.resize(temp_img,(image_dims[0],image_dims[1]))

    im = img_to_array(temp_img)

    im = np.expand_dims(im, axis=0)

    im = preprocess_input(im)

    output.append([img,dummy_cols[np.argmax(model.predict(im))]])

  return output

    
args['model']='vgg16_5.hdf5'

output_label=prediction(model_path=args['model'])

# # # load_model()

pd.DataFrame(output_label, columns=['Image','Class']).to_csv('sub13.csv',index=False)
