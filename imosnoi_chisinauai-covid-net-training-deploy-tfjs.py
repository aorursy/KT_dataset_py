# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2, glob, random

import matplotlib.pyplot as plt #for plotting things

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

# Any results you write to the current directory are saved as output.
!ls /kaggle/input/covidchestxraydataset/images
import keras

from keras import backend as K

from keras.layers.core import Dense, Activation

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import Model

from keras.applications import imagenet_utils

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input

import numpy as np

from IPython.display import Image

from keras.optimizers import Adam
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.



x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3

preds=Dense(4,activation='softmax')(x) 
model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:

    layer.trainable=False

# or if we want to set the first 20 layers of the network to be non-trainable

for layer in model.layers[:20]:

    layer.trainable=False

for layer in model.layers[20:]:

    layer.trainable=True



datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

def preprocessI(path):

            img = cv2.imread(path)

            #print(img)

            im = cv2.resize(img,(324,324))

            im = datagen.apply_transform(im,{

                'theta':np.random.randint(0,40)-20,

                'tx':np.random.rand()/4,

                'ty':np.random.rand()/4,

                'flip_horizontal':np.random.rand()>0.5

            })

            im = cv2.resize(im,(224,224))

            return im

def getGenerators(imgs_covid, nr=1, folder='train'):

    foldere = glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/'+folder+'/*') 

    while 1:

        x, y = [], []

        #nr = 1

        for i, clasa in enumerate(foldere):

            imgs = glob.glob(clasa+'/*')

            y_ = np.zeros(4)

            y_[i]=1

            #print(clasa)

            for j in range(nr):

                #path = imgs[j]#

                path = random.choice(imgs)

                #print(path)

                im = preprocessI(path)

                x.append(im)            

                y.append(y_)

                #print(im.shape)

        y_ = np.zeros(4)

        y_[3] = 1 

        for j in range(nr):

                #path = imgs_covid[j]#

                path = random.choice(imgs_covid)

                #print(path)

                im = preprocessI(path)

                x.append(im)

                y.append(y_)

        x = keras.applications.mobilenet.preprocess_input(np.array(x))

        yield x, y

covid_img = glob.glob('/kaggle/input/covidchestxraydataset/images/*')

gn_train = getGenerators(covid_img[:60], nr=400, folder='train')

gn_val = getGenerators(covid_img[60:120], nr=100, folder='val')

gn_test = getGenerators(covid_img[120:], nr=100, folder='test')

x_tr,y_tr = next(gn_train)

print('val')

x_val,y_val = next(gn_val)

print('test')

x_test,y_test = next(gn_test)

#plt.imshow(x_test[-1]), len(covid_img)
y_tr = np.array(y_tr)

y_val = np.array(y_val)

y_test = np.array(y_test)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

hi = model.fit(x_tr,y_tr, epochs=15, validation_data = (x_val, y_val), batch_size=29)

acc = model.evaluate(x_test, y_test)

acc


y_true = np.argmax(y_test,axis=1)

y_ = model.predict(x_test)

y_pred = np.argmax(y_, axis=1)

confusion_matrix(y_true, y_pred)
y_true, y_pred


target_names = ['normal', 'pneumonia', 'covid']

print(classification_report(y_true, y_pred, target_names=target_names))
model.save('model.h5')
!pip install tensorflowjs
!tensorflowjs_converter --input_format=keras model.h5 tfjs_model
!ls
!tar -czvf model.tar.gz tfjs_model