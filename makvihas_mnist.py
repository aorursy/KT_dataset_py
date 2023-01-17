import tensorflow as tf

import tensorflow.contrib.layers as layers

import numpy as np

import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/mnist", one_hot = True) #load mnist

from skimage.transform import *

from skimage.util import *
## Folowing functions are for data augmentation

def random_crop(img):

    x=np.pad(img.reshape(28,28),(5,5),mode='constant') #pads (28,28) image with pad_width 5. So OUTOUT_SHAPE=(38,38)   

    a=random.randint(0,10) #random integer between (0,OUTPUT_SHAPE-input_shape) i.e. 38-28 = 10

    b=random.randint(0,10) #random integer between (0,OUTPUT_SHAPE-input_shape) i.e. 38-28 = 10

    return x[a:a+28,b:b+28]#randomly selects (28,28) patch from (38,38) padded image 

def random_rot(img,angle=30):

    ang=np.random.uniform(-30,30,1) #randomly selects angle from -30° to +30°

    return rotate(img.reshape(28,28),ang) #rotates image at selected angle

def rand_noise(img):

    return img.reshape(28,28)+np.random.normal(0,0.05,size=(28,28)) #adds random gaussian noise

train_=list(mnist.train.images.reshape(-1,28,28))

labels_=list(mnist.train.labels)
for i in [random_crop,random_rot]:

    a=random.sample(range(0,55000),22500*2) #randomly selects 45000 integers from (0,len(train_)) range

    #looping through selected integers

    for j in a:

        train_.append(i(mnist.train.images[j])) #applying function and appending to training set

        labels_.append(mnist.train.labels[j])
#Main Tensorflow model

def model(x):

    x =  layers.conv2d(x,32*2,(3,3),padding='VALID',activation_fn=None) #conv2d from tensorflow.contrib.layers

    x =  tf.nn.relu(layers.batch_norm(x))                               #batch_norm from tensorflow.contrib.layers

    x =  layers.conv2d(x,64*2,(3,3),padding='VALID',activation_fn=None) 

    x =  tf.nn.relu(layers.batch_norm(x))

    x =  layers.max_pool2d(x,2)                                        #max_pool2d from tensorflow.contrib.layers

  

    x =  layers.conv2d(x,64*2,(3,3),padding='SAME',activation_fn=None)

    x =  tf.nn.relu(layers.batch_norm(x))

    x =  layers.conv2d(x,128*2,(3,3),padding='SAME',activation_fn=None)

    x =  tf.nn.relu(layers.batch_norm(x))

  

    x =  layers.max_pool2d(x,2)

  

    x =  layers.conv2d(x,128*2,(3,3),padding='VALID',activation_fn=None)

    x =  tf.nn.relu(layers.batch_norm(x))

    x =  layers.conv2d(x,256*2,(3,3),padding='VALID',activation_fn=None)

    x =  tf.nn.relu(layers.batch_norm(x))

      

    x =  layers.max_pool2d(x,2)

   

    x =  layers.flatten(x) #flattening

   

    x =  layers.fully_connected(x,64,activation_fn=None)  #fully_connected from tensorflow.contrib.layers

    x =  tf.nn.relu(layers.batch_norm(x))

  

    x =  layers.fully_connected(x,10,activation_fn=None)  

    x =  tf.nn.softmax(layers.batch_norm(x))

    return x                                              #predicted output tensor of shape (batch_size,10)



#batch size

batch=200

#create placeholders

x = tf.placeholder(tf.float32, shape=(None, 28, 28,1))

y = tf.placeholder(tf.int32,shape=(None,10))

# y_=tf.one_hot(y,10)[:,-1,:]

#I used 9 CNNs

models=[model(x) for _ in range(9)]

y_pred=tf.reduce_mean(models,0) #reduce_sum at axis=0 and output shape is (batch_size,10)



loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(y,logits=y_pred))

opt=tf.train.AdamOptimizer().minimize(loss)



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
sess=tf.Session()

sess.run(tf.global_variables_initializer())
epoch=0

k=0

batch=200

for i in range(int(len(train_)/batch*15)):

    if k>=len(train_):

        epoch+=1

        k=0

    x_=np.array(train_[k:k+batch])

    y_=np.array(labels_[k:k+batch])

    sess.run(opt,{x:x_.reshape(-1,28,28,1),y:y_})

    k+=batch

    if i%72.5==0:

        print("EPOCH: ",i/725)
import pandas as pd

csv=pd.read_csv("../input/test.csv")

test=[list(csv.iloc[i]) for i in range(len(csv))]

label=[]

for i in range(140):

    label.extend(np.argmax(sess.run(y_pred,{x:np.reshape(test[i*200:(i+1)*200],(-1,28,28,1))/255}),1))
csv=pd.DataFrame({'ImageId':range(1,28001),'Label':label})

csv.to_csv("submission.csv",index=False)