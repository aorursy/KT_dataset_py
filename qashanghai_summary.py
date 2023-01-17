# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import cv2

from tensorflow.keras import layers

import matplotlib.pyplot as plt

import skimage.draw as skdraw
import re,sys



img_path = (

'/kaggle/input/tusimple-images1/imgs_0.npy',

'/kaggle/input/tusimple-images1/imgs_1.npy',

'/kaggle/input/tusimple-images2/imgs_2.npy',

'/kaggle/input/tusimple-images2/imgs_3.npy',

'/kaggle/input/tusimple-images3/imgs_4.npy',

'/kaggle/input/tusimple-images3/imgs_5.npy',

'/kaggle/input/tusimple-images4/imgs_6.npy',

'/kaggle/input/tusimple-images4/imgs_7.npy',

)



img_m,img_n,_ = np.load('/kaggle/input/tusimple-images1/imgs_0.npy')[0].shape
samples = np.load('/kaggle/input/tusimple-annotation/samples.npy')

lane_marks = np.load('/kaggle/input/tusimple-annotation/lanes.npy')



def img_set(i):

    return np.load(img_path[i])



def preprosses_batch(imgs):

    return imgs[:,::4,::4,:][:,:160,:320]



def preprosses(imgs):

    return imgs[::4,::4,:][:160,:320]



def lane_mask(k):

    mark = lane_marks[k]

    res = np.zeros((img_m,img_n,3))

    poly = [None]*4

    

    for i in range(4):

        if (mark[i]==-2).all() and i>=2:

            mark[i] = mark[i-2]

        if (mark[i]==-2).all() and i<2:

            mark[i] = mark[1-i]

        

        X = samples[mark[i]!=-2]

        Y = mark[i][mark[i]!=-2]

        #print(X,Y)

        z = np.polyfit(X,Y,2)

        poly[i] = np.poly1d(z)



    for ii,i in enumerate((2,0,1)):

        s = np.concatenate([samples,[800]],0)

        p = np.zeros(((len(s))*2,2))

        p[:len(s),0] = s

        p[len(s):,0] = np.flip(s)

        p[:len(s),1] = poly[i](s)

        p[len(s):,1] = poly[(2,0,1,3)[ii+1]](np.flip(s))

        t = np.concatenate([mark[i],[0,0],mark[(2,0,1,3)[ii+1]]],0)

        p = p[t!=-2]

        res[:,:,i] = skdraw.polygon2mask([img_m,img_n],p)

    return preprosses(res*1)



def predict(model,i):

    result = model.predict(img[i:i+1], batch_size=32)[0]



    a,b,c = np.gradient(result)[0],np.gradient(result)[1],np.gradient(result)[2]

    res = np.zeros((160,320,4))

    res[:,:,1:]=result

    res[:,:,0] = 0.2

    return result,(img[i]+result),(res.argmax(2))
lst = []

for i in range(8):

    ig = img_set(i)

    ig = preprosses_batch(ig)

    lst.append(ig)

    del ig

img = np.concatenate(lst,0)[:1500]

img = ((img<0)*256+img)/256

del lst
'''mask = np.zeros((len(lane_marks),160,320,3),dtype = np.int8)

for i in range(len(lane_marks)):

    print(i,end=' ')

    mask[i] = lane_mask(i)

    if i%100 == 0:

        np.save('mask.npy',mask)

np.save('mask.npy',mask)'''
mask = np.load("/kaggle/input/tusimple-mask/mask.npy")[:1500]
input_x = tf.keras.Input(shape=(160,320,3))

up = tf.keras.layers.UpSampling2D

down = tf.keras.layers.MaxPooling2D



c = lambda cha: tf.keras.layers.Conv2D(cha, 2, 

                                       activation='relu', 

                                       padding="same",

                                       #kernel_initializer='random_normal'

                                       use_bias=True,

                                       bias_initializer = "zeros"

                                      )



c11 = c(64)(input_x)

c12 = down((2,2))(c(64)(c11))



c21 = (c(128)(c12))

c22 = down((2,2))(c(128)(c21))



c31 = ((c(256)(c22)))

c32 = ((c(256)(c31)))

c33 = down((2,2))(c(256)(c32))



c41 = ((c(512)(c33)))

c42 = ((c(512)(c41)))

c43 = down((2,2))(c(512)(c42))



c51 = ((c(512)(c43)))

c52 = ((c(512)(c51)))

c53 = (c(512)(c52))



u3 = c(256)(up((4,4))(c53))

u4 = c(128)(up((2,2))(u3+c31))

u5 = c(64)(up((2,2))(u4+c21))

u6 = c(3)(u5+c11)



y = u6
dataset = tf.data.Dataset.from_tensor_slices((img[:1000], mask[:1000]))

dataset = dataset.batch(32)

dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((img[1000:], mask[1000:]))

val_dataset = val_dataset.batch(32)

val_dataset = val_dataset.repeat()
model = tf.keras.Model(inputs=input_x, outputs=y)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0000001),

             loss='mean_squared_error',

             metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(dataset, epochs=30, steps_per_epoch=20,

          validation_data=val_dataset, validation_steps=3)
a,b,c = predict(model,1410)

plt.imshow(b)
model.save('fcn')
def Hough(e,m,n):

    a,b = e.shape

    res = np.zeros((m,n))

    x,y = np.where(e>0)

    for i in range(len(x)):

        xx,yy = x[i],y[i]

        r = np.sqrt(xx**2+yy**2)

        for i,theta in enumerate(np.linspace(0,2*np.pi,m)):

            res[i,int((xx*np.cos(theta)+yy*np.sin(theta))/np.sqrt(a**2+b**2)*n)]+=1

    return res