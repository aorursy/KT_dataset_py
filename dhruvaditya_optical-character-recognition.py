# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
import string
# from ../input/ctcmodel/CTCModel/ import CTCModel

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os,fnmatch
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Only Selecting the Images and excluding the directories in the folder
# images_names=fnmatch.filter(os.listdir('/kaggle/input/captcha-version-2-images/samples/'),'*.*')
from PIL import Image
# from matplotlib.pyplot import imshow
# images_names.sort()
# print(images_names)
# img = Image.open('../input/linesdata/data/data/sentences/s01/s01-000/'+images_names[0],'r')
# img = img.resize((784,32), Image.ANTIALIAS)
# plt.imshow(img)
# img=np.asarray(img)
# img=img[:,:,0]
# print(img.shape)

text_file=open(r"../input/linesdata/UsedSentences.txt","r")
details=[]
outputs=[]
names=[]
for line in text_file:
    a=line.split('#')
    outputs.append(a[1].strip('\n'))
    details.append(a[0])

for detail in details:
    a=detail.split(' ')
    names.append(a[0])
X=[]
for name in names:
    img=Image.open('../input/linesdata/data/data/sentences/s01/s01-000/'+name+'.png','r')
    img = img.resize((784,32), Image.ANTIALIAS)
    img=np.asarray(img)
    img=img[:,:,0]
    X.append(img)

X=np.asarray(X)
plt.imshow(X[42])
plt.title(outputs[42])
print("No of Images :",X.shape[0])

symbols = " "+string.ascii_lowercase + string.ascii_uppercase+"0123456789.,*&!@~():`^]¢‘;|-"
print("Characters :",symbols)
print("No of chars :",len(symbols))

# print(os.listdir('../input/linesdata/data/sentences/s01/'))

# Y=[]
# indices=[]
# values=[]
# print(len(outputs))
# for example_no,name in enumerate(outputs):
#     for letter_no,letter in enumerate(name): 
#         indices.append([example_no,letter_no])
#         values.append(symbols.index(letter))

# print(indices)
# sparce_y=tf.SparseTensor(indices=indices, values=values, dense_shape=(len(outputs),7))
# print(sparce_y.shape)'
# print(outputs[0])
Y=np.zeros(shape=(len(outputs),98,len(symbols)))
for example_no,name in enumerate(outputs):
    for letter_no,letter in enumerate(name):
        try:
            Y[example_no][letter_no][symbols.index(letter)]=1
        except:
            print(letter,end=" ")
# print(Y.shape)
# print(Y[0])
# print(values)
# train_X=X[:int(0.8*len(X))]
# train_Y=y[:,:int(0.8*len(X)),:]
# train_outputs=outputs[:int(0.8*len(X))]

# validation_X=X[int(0.8*len(X)):int(0.9*len(X))]
# validation_Y=y[:,int(0.8*len(X)):int(0.9*len(X)),:]
# validation_outputs=outputs[int(0.8*len(X)):int(0.9*len(X))]

# test_X=X[int(0.9*len(X)):]
# test_Y=y[:,int(0.9*len(X)):,:]
# test_outputs=outputs[int(0.9*len(X)):]


# print("Train X Shape",train_X.shape)
# print("Train Y Shape",train_Y.shape)

# print("Validation X Shape",validation_X.shape)
# print("Validation Y Shape",validation_Y.shape)

# print("Test X Shape",test_X.shape)
# print("Test Y Shape",test_Y.shape)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
print(X.shape)
# Neural Network Model 
# Try Removing Batch Normalisation and see how the performance decreases.

image=keras.layers.Input((32,784,1))
conv1=keras.layers.Conv2D(16,(3,3),activation='relu',padding='same')(image)
mp1=keras.layers.MaxPooling2D((2,2),padding='same')(conv1)
conv2=keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(mp1)
mp2=keras.layers.MaxPooling2D((2,2),padding='same')(conv2)
conv3=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(mp2)
mp3=keras.layers.MaxPooling2D((2,2),padding='same')(conv3)
conv4=keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(mp3)
mp4=keras.layers.MaxPooling2D((2,1),padding='same')(conv4)
conv5=keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(mp4)
mp5=keras.layers.MaxPooling2D((2,1),padding='same')(conv5)
conv6=keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(mp5)
# mp6=keras.layers.MaxPooling2D((1,3),padding='same')(conv6)
bn=keras.layers.BatchNormalization()(conv6)
sq=keras.backend.squeeze(bn,axis=1)

rn1=keras.layers.Bidirectional(keras.layers.LSTM(256,return_sequences=True))(sq)
rn2=keras.layers.Bidirectional(keras.layers.LSTM(256,return_sequences=True))(rn1)

exd=keras.backend.expand_dims(rn2,axis=2)
maping=keras.layers.Conv2D(len(symbols),(2,2),activation='relu',padding='same')(exd)
maping=keras.backend.squeeze(maping,axis=2)
maping = tf.keras.layers.Softmax()(maping)

# bn = keras.layers.BatchNormalization()(conv3)
model=keras.Model(image,maping)
# model.compile(loss=tf.nn.ctc_loss(labels=sparce_y,logits=X,label_length=7,logit_length=7,blank_index=7))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X,Y,epochs=15)
# a=tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
pred=model.predict(X)
from random import *
times=100
for i in range(times):
    exp=randint(0,pred.shape[0]-1)
    index=randint(0,len(symbols))

    print(np.argmax(pred[exp][index]),np.argmax(Y[exp][index]))
    
index=91
c=""
for i in range(len(pred[0])):
    c=c+(symbols[np.argmax(pred[index][i])])
print("predicted:",c.strip())
print("\nOrignal:",outputs[index])
plt.imshow(X[index][:,:,0])

