## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



from PIL import Image



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#os.listdir("../input/all-dogs/all-dogs/")

#os.listdir("../input/annotation/Annotation/")
im = Image.open("../input/all-dogs/all-dogs/"+os.listdir("../input/all-dogs/all-dogs/")[100])

im=im.resize((64, 64 ))

np_im = np.array(im)

print(np_im.shape)
import matplotlib.pyplot as plt

plt.imshow(np_im)

plt.show()
def makeMasterArray(dim):

    masterLst=[]

    for pic in os.listdir("../input/all-dogs/all-dogs/"):

        im = Image.open("../input/all-dogs/all-dogs/"+pic)

        im=im.resize((dim, dim))

        np_im = np.array(im)

        masterLst.append(np_im)

    return np.array(masterLst)
masterDogArray=makeMasterArray(64)
def conv(x,w,b,stride,name):

    with tf.variable_scope('conv'):

        return tf.nn.conv2d(x,filter=w,strides=[1,stride,stride,1],padding='SAME',name=name)+b



def deconv(x,w,b,shape,stride,name):

    with tf.variable_scope('deconv'):

        return tf.nn.conv2d_transpose(x,filter=w,output_shape=shape,strides=[1,stride,stride,1],padding='SAME',name=name)+ b



def lrelu(x,alpha=0.2):

    with tf.variable_scope('leakyReLU'):

        return tf.maximum(x,alpha*x)



def generator(X,batch_size=64):

    

    with tf.variable_scope('generator'):

        K=256

        L=128

        M=64

        

        W1=tf.get_variable('G_W1',[100,16*16*K],initializer=tf.random_normal_initializer(stddev=0.1))

        B1=tf.get_variable('G_B1',[16*16*K],initializer=tf.constant_initializer())

        

        W2=tf.get_variable('G_W2',[4,4,M,K],initializer=tf.random_normal_initializer(stddev=0.1))

        B2=tf.get_variable('G_B2',[M],initializer=tf.constant_initializer())

        

        W3=tf.get_variable('G_W3',[4,4,3,M],initializer=tf.random_normal_initializer(stddev=0.1))

        B3=tf.get_variable('G_B3',[3],initializer=tf.constant_initializer())

        

        X=lrelu(tf.matmul(X,W1)+B1)

        X=tf.reshape(X,[batch_size,16,16,K])

        

        deconv1=deconv(X,W2,B2,shape=[batch_size,32,32,M],stride=2,name='deconv1')

        

        bn1=tf.contrib.layers.batch_norm(deconv1)

        

        deconv22=deconv(tf.nn.dropout(lrelu(bn1),0.4),W3,B3,shape=[batch_size,64,64,3],stride=2,name='deconv2')

        

        return deconv22



def discriminator(X,reuse=False):

    with tf.variable_scope('discriminator'):

        if reuse:

            tf.get_variable_scope().reuse_variables()

        K=64

        M=128

        N=256

        

        W1=tf.get_variable('D_W1',[4,4,3,K],initializer=tf.random_normal_initializer(stddev=0.1))

        B1=tf.get_variable('D_B1',[K],initializer=tf.constant_initializer())

        

        W2=tf.get_variable('D_W2',[4,4,K,M],initializer=tf.random_normal_initializer(stddev=0.1))

        B2=tf.get_variable('D_B2',[M],initializer=tf.constant_initializer())

        

        W3=tf.get_variable('D_W3',[16*16*M,N],initializer=tf.random_normal_initializer(stddev=0.1))

        B3=tf.get_variable('D_B3',[N],initializer=tf.constant_initializer())

        

        W4=tf.get_variable('D_W4',[N,1],initializer=tf.random_normal_initializer(stddev=0.1))

        B4=tf.get_variable('D_B4',[1],initializer=tf.constant_initializer())

        

        X=tf.reshape(X,[-1,64,64,3],'reshape')

        conv1=conv(X,W1,B1,stride=2,name='conv1')

        bn1=tf.contrib.layers.batch_norm(conv1)

        

        conv2=conv(tf.nn.dropout(lrelu(bn1),0.4),W2,B2,stride=2,name='conv2')

        bn2=tf.contrib.layers.batch_norm(conv2)

        

        flat=tf.reshape(tf.nn.dropout(lrelu(bn2),0.4),[-1,16*16*M],name='flat')

        dense=lrelu(tf.matmul(flat,W3)+B3)

        logits=tf.matmul(dense,W4)+B4

        prob=tf.nn.sigmoid(logits)



        return prob,logits

        


def train(batch_size):

    with tf.variable_scope('placeholder'):

        X=tf.placeholder(tf.float32,[None,64,64,3])

        z=tf.placeholder(tf.float32,[None,100])

        G=generator(z,batch_size)

        D_real,D_real_logits=discriminator(X,reuse=False)

        D_fake,D_fake_logits=discriminator(G,reuse=True)

    

    with tf.variable_scope('D_loss'):

        d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,labels=tf.ones_like(D_real_logits)))

        d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,labels=tf.zeros_like(D_fake_logits)))

        d_loss=d_loss_real+d_loss_fake

    

    with tf.variable_scope('G_loss'):

        g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,labels=tf.ones_like(D_fake_logits)))

        tvar=tf.trainable_variables()

        dvar=[var for var in tvar if 'discriminator' in var.name]

        gvar=[var for var in tvar if 'generator' in var.name]

        

        

    with tf.variable_scope('train'):

        d_train_step=tf.train.AdamOptimizer().minimize(d_loss,var_list=dvar)

        g_train_step=tf.train.AdamOptimizer().minimize(g_loss,var_list=gvar)

        

    sess=tf.Session()

    init=tf.global_variables_initializer()

    sess.run(init)



    for iters in range(30):

        for batchIdx in range(int(masterDogArray.shape[0]/batch_size)):

            batch_X=masterDogArray[batchIdx*batch_size:(batchIdx+1)*batch_size]

            batch_noise=np.random.uniform(-1.0,1.0,[batch_size,100])

            _,d_loss_print=sess.run([d_train_step,d_loss],feed_dict={X:batch_X,z:batch_noise})

            _,g_loss_print=sess.run([g_train_step,g_loss],feed_dict={z:batch_noise})

            

            #print('g_loss: %f, d_loss: %f' %(g_loss_print,d_loss_print))

        samples=sess.run(G,feed_dict={z:np.random.uniform(-1.0,1.0,[batch_size,100])})



        plt.imshow(samples[0])

        plt.show()

            


train(64)
