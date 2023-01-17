# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from PIL import Image

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pikaFolder=os.listdir('/kaggle/input/complete-pokemon-image-dataset/pokemon/')

allPikaImage=[]

for indivPika in pikaFolder:

    allImages=os.listdir('/kaggle/input/complete-pokemon-image-dataset/pokemon/'+indivPika+'/')

    for indivImg in allImages:

        img=Image.open('/kaggle/input/complete-pokemon-image-dataset/pokemon/'+indivPika+'/'+indivImg)

        img=img.resize((128,128), Image.ANTIALIAS)

        img=np.array(img)

        allPikaImage.append(img)
allPikaImage=np.array(allPikaImage)/255.0
import tensorflow as tf

def make_encoder(data):

    conv0 = tf.layers.conv2d(inputs=data,filters=64,

      kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

    pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=2)

    

    conv1 = tf.layers.conv2d(inputs=pool0,filters=128,

      kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    

    conv2 = tf.layers.conv2d(inputs=pool1,filters=256,

      kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    

    conv3 = tf.layers.conv2d(inputs=pool2,filters=256,

      kernel_size=[2, 2],padding="same",activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    

    conv4 = tf.layers.conv2d(inputs=pool3,filters=512,

      kernel_size=[2, 2],padding="same",activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    

    #conv5 = tf.layers.conv2d(inputs=pool4,filters=32,

    #  kernel_size=[2, 2],padding="same",activation=tf.nn.relu)

    

    flatpool4=tf.reshape(pool4,[-1,4*4*512])

    flatpool5=tf.layers.dense(flatpool4,512)

    encoded = tf.layers.dense(flatpool5, 64)

    #return conv5

    return encoded



def make_decoder(data):

    

    #deConvFilterORIG = tf.get_variable('filterORIG', shape = [2, 2, 512, 32], dtype = tf.float32)

    #deconv0 = tf.nn.conv2d_transpose(data,filter = deConvFilterORIG,

    #                          output_shape =tf.stack([tf.shape(data)[0], 4, 4, 512]),

    #                          strides = [1, 1, 1, 1], padding = 'SAME')

    x1=tf.layers.dense(data ,512)

    x=tf.layers.dense(x1,4*4*512)

    deconv0=tf.reshape(x,[-1,4,4,512])

    

    deConvFilter0 = tf.get_variable('filter0', shape = [2, 2, 256, 512], dtype = tf.float32)

    deconv1 = tf.nn.conv2d_transpose(deconv0,filter = deConvFilter0,

                              output_shape =tf.stack([tf.shape(data)[0], 8, 8, 256]),

                              strides = [1, 2, 2, 1], padding = 'SAME')

    

    

    deConvFilter1 = tf.get_variable('filter1', shape = [2, 2, 256, 256], dtype = tf.float32)

    deconv2 = tf.nn.conv2d_transpose(deconv1,filter = deConvFilter1,

                              output_shape =tf.stack([tf.shape(data)[0], 16, 16, 256]),

                              strides = [1, 2, 2, 1], padding = 'SAME')

    deconv2=tf.nn.relu(deconv2)

    

    deConvFilter2 = tf.get_variable('filter2', shape = [3, 3, 128, 256], dtype = tf.float32)

    deconv3 = tf.nn.conv2d_transpose(deconv2,filter = deConvFilter2,

                              output_shape =  tf.stack([tf.shape(data)[0], 32, 32, 128]),

                              strides = [1, 2, 2, 1], padding = 'SAME')

    deconv3=tf.nn.relu(deconv3)

    

    deConvFilter3 = tf.get_variable('filter3', shape = [3, 3, 64, 128], dtype = tf.float32)

    deconv4 = tf.nn.conv2d_transpose(deconv3,filter = deConvFilter3,

                              output_shape =  tf.stack([tf.shape(data)[0],64, 64, 64]),

                              strides = [1, 2, 2, 1], padding = 'SAME')

    

    

    deConvFilter4 = tf.get_variable('filter4', shape = [3, 3, 64, 64], dtype = tf.float32)

    deconv5 = tf.nn.conv2d_transpose(deconv4,filter = deConvFilter4,

                              output_shape =  tf.stack([tf.shape(data)[0],128, 128, 64]),

                              strides = [1, 2, 2, 1], padding = 'SAME')

    

    

    deConvFilter5 = tf.get_variable('filter5', shape = [3, 3, 3, 64], dtype = tf.float32)

    decoded = tf.nn.conv2d_transpose(deconv5,filter = deConvFilter5,

                              output_shape =  tf.stack([tf.shape(data)[0],128, 128, 3]),

                              strides = [1, 1, 1, 1], padding = 'SAME')

    

    

    

    return tf.nn.relu(decoded)


make_encoder = tf.make_template('encoder', make_encoder)

make_decoder = tf.make_template('decoder', make_decoder)

inputPika = tf.placeholder(tf.float32, [None,128,128,3])



hidden = make_encoder(inputPika)

recon = make_decoder(hidden)

loss = tf.nn.l2_loss(recon - inputPika)  # L2 loss

training = tf.train.AdamOptimizer(0.0001).minimize(loss)



sess=tf.Session()

init=tf.global_variables_initializer()

sess.run(init)

batchSize=64

avgLoss=0.0



for i in range(1):

    for epoch in range(70):

        avgLoss=0.0

        for idx in range(int(allPikaImage.shape[0]/batchSize)-1):

            temp=allPikaImage[idx*batchSize:(idx+1)*batchSize]

            err,_=sess.run([loss,training],feed_dict={inputPika:temp})

            avgLoss+=err

        print("For epoch "+str(i*100+epoch+1)+" the error is: "+str(avgLoss/int(allPikaImage.shape[0]/batchSize)-1))
evalPika = tf.placeholder(tf.float32, [None,128,128,3])



latent = make_encoder(evalPika)



batchSize=1

avgPika=None



for idx in range(1,2):

#for idx in range(int(allPikaImage.shape[0]/batchSize)-1):

    temp=allPikaImage[idx*batchSize:(idx+1)*batchSize]

    narrowPika=sess.run([latent],feed_dict={evalPika:temp})

    if avgPika is None:

        avgPika=np.array(narrowPika).mean(axis=1)[0]

    else:

        avgPika+=np.array(narrowPika).mean(axis=1)[0]

#avgPika=avgPika/(int(allPikaImage.shape[0]/batchSize)-1)

avgPika=avgPika/1
newNarrowPika=np.array([avgPika])

zippedPika=tf.placeholder(tf.float32,[None,64])

reconPika = make_decoder(zippedPika)

avgImg=sess.run(reconPika,feed_dict={zippedPika:newNarrowPika})

plt.imshow(avgImg[0])