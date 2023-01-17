import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Activation,Dropout,Dense,Conv2D,AveragePooling2D,Flatten,ZeroPadding2D,MaxPooling2D,Convolution2D,MaxPooling2D

from keras import optimizers

from sklearn.model_selection import train_test_split

from keras.models import Sequential

import seaborn as sns

from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

import math

import cv2

import imageio

from os import listdir

import warnings

import filecmp

from PIL import Image

import numpy as np

import pandas as pd

import os

import cv2

import PIL

import gc

import psutil

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import set_random_seed

from tqdm import tqdm

from math import ceil

import math

import sys

import gc



import keras

from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

from keras.models import Sequential, Model

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input

from keras.layers import Dropout, Flatten, Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.activations import softmax, relu, elu

from keras.optimizers import Adam, rmsprop, RMSprop   ,SGD

from keras.layers import BatchNormalization, LeakyReLU

from tqdm import tqdm
listdir('../input/ucla-protest-dataset/test/test/')

basepath1='../input/ucla-protest-dataset/train1/train1/'

basepath2='../input/ucla-protest-dataset/train2/train2/'

basepath3='../input/ucla-protest-dataset/train3/train3/'

testbase='../input/ucla-protest-dataset/test/test/'

annot_train=pd.read_csv('../input/ucla-protest-dataset/annot_train - annot_train.csv')

annot_test=pd.read_csv('../input/ucla-protest-dataset/test csv.csv')
annot_train=annot_train.replace("-",0)

annot_test=annot_test.replace("-",0)
from time import ctime

import sys

print(ctime())

train=[]

test=[]

label=[]

label2=[]

i=0

j=0

for file in listdir(basepath1):

    filepath=basepath1+file

    if(file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') ):

        if(annot_train[annot_train.fname==file]['protest'].all()==0 and j<9300):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            j+=1

            label.append(0)

        elif(annot_train[annot_train.fname==file]['protest'].all()==1):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            label.append(1)

        else:

            i+=1

            

print(ctime())

for file in listdir(basepath2):

    filepath=basepath2+file

    if(file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') ):

        if(annot_train[annot_train.fname==file]['protest'].all()==0 and j<9300):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            j+=1

            label.append(0)

        elif(annot_train[annot_train.fname==file]['protest'].all()==1):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            label.append(1)

        else:

            i+=1

        

print(ctime())



for file in listdir(basepath3):

    filepath=basepath3+file

    if(file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') ):

        if(annot_train[annot_train.fname==file]['violence'].all()==0 and j<9300):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            j+=1

            label.append(0)

        elif(annot_train[annot_train.fname==file]['protest'].all()==1):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            train.append(image)

            i+=1

            label.append(1)

        else:

            i+=1

        

print(ctime())

for file in listdir(testbase):

    filepath=testbase+file

    if(file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') ):

        if(annot_test[annot_test.fname==file]['protest'].all()==0):

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            test.append(image)

            label2.append(0)

        else:

            image = imageio.imread(filepath)

            image=cv2.resize(image,(50,50))

            image=image/255

            test.append(image)

            label2.append(1)

            

        

print(ctime())

def create_resnet(img_dim, CHANNEL, n_class):

    input_tensor = Input(shape=(img_dim, img_dim, CHANNEL))



    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    base_model.load_weights('../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

#     base_model.load_weights('../input/restnet101/resnet101_weights_tf.h5')



    



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.4)(x)

    x = Dense(2048, activation=elu)(x)

    x = Dropout(0.4)(x)

    x = BatchNormalization()(x)

    x = Dense(1024, activation=elu)(x)

    x = Dropout(0.4)(x)

    x = BatchNormalization()(x)

    x = Dense(512, activation=elu)(x)

    x = Dropout(0.4)(x)

    x = BatchNormalization()(x)

    output_layer = Dense(n_class, activation='sigmoid', name="Output_Layer")(x)

    model_resnet = Model(input_tensor, output_layer)



    return model_resnet



model_resnet = create_resnet(50, 3, 1)
train=np.array(train)

label=np.array(label)

label=label.reshape(label.shape[0],1)
es=EarlyStopping(monitor='val_acc',mode=max,verbose=1,patience=3)

mc=ModelCheckpoint('best_model.h5',monitor='val_acc',mode=max,verbose=1,save_best_only=True)
lr = 1e-3

optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=lr, decay=0.01) 

model_resnet.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

# model.summary()

gc.collect()
history=model_resnet.fit(train,label,epochs=10,batch_size=10,verbose=1,validation_split=0.35,shuffle=True,callbacks=[es,mc])
model=keras.models.load_model('best_model.h5')
accu = history.history['acc']

val_acc = history.history['val_acc']



plt.plot(accu, label="Accuracy")

plt.plot(val_acc)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Acc', 'val_acc'])

plt.plot(np.argmax(history.history["val_acc"]), np.max(history.history["val_acc"]), marker="x", color="r",

         label="best model")

plt.show()
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(history.history["loss"], label="loss")

plt.plot(history.history["val_loss"], label="val_loss")

plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",

         label="best model")

plt.xlabel("Epochs")

plt.ylabel("Binary CrossEntropy")

plt.legend();
test=np.array(test)



predict=model.predict(test)

predict=np.squeeze(predict,axis=1)

for i in range(predict.shape[0]):

    if(predict[i]<0.5):

        predict[i]=0

    else:

        predict[i]=1





label2=np.array(label2)

print("Accuracy: ",round(accuracy_score(predict,label2)*100,2))

model.save('besttt_model.h5')