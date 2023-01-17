# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mnistO = pd.read_csv("../input/train.csv")
mnist = mnistO[mnistO.iloc[:,0]==5]
mnist = mnist.iloc[:,1:]
mnist = np.array(mnist)/255
mnist.shape
plt.imshow(np.reshape(mnist[0],(28,28)))
def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        l = tf.layers.dense(inputs = z, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.dense(inputs = l, units = 784, activation=tf.nn.tanh)
        return l
def discriminator(x, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        l = tf.layers.dense(inputs = x, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.batch_normalization(l)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.dense(inputs = l, units = 128, activation=tf.nn.leaky_relu)
        l = tf.layers.dense(inputs = l, units = 1)
        print(l)
        return l
tf.reset_default_graph()
real_images = tf.placeholder(tf.float32,shape = [None, 784])
z = tf.placeholder(tf.float32, shape = [None, 100])
G = generator(z)
DLogitsReal = discriminator(real_images)
DLogitsFake = discriminator(G, reuse=True)
def loss(logits, labels):
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    return error
DRealLoss = loss(DLogitsReal, tf.ones_like(DLogitsReal))
DFakeLoss = loss(DLogitsFake, tf.zeros_like(DLogitsFake))
DLoss=DRealLoss+DFakeLoss
GLoss = loss(DLogitsFake, tf.ones_like(DLogitsFake))
lr=0.0001
TVars=tf.trainable_variables()
DVars=[var for var in TVars if 'dis' in var.name]
GVars=[var for var in TVars if 'gen' in var.name]
D_trainer=tf.train.AdamOptimizer(lr).minimize(DLoss,var_list=DVars)
G_trainer=tf.train.AdamOptimizer(lr).minimize(GLoss,var_list=GVars)
batch_size=50
epochs=100+1
init=tf.global_variables_initializer()
samples=[] #generator examples
sess = tf.Session()
sess.run(init)
Dsum = []
Gsum = []
for epoch in range(epochs):
    num_batches = len(mnist)//batch_size
    for i in range(num_batches):
        batch = mnist[ i * batch_size : (( i + 1 ) * batch_size) ]
        batch_images=batch
        batch_images=batch_images
        batch_z=np.random.uniform(-1,1,size=(batch_size,100))
    
        Dl, _=sess.run([DLoss, D_trainer],feed_dict={real_images:batch_images,z:batch_z})
        Gl, _=sess.run([GLoss, G_trainer],feed_dict={z:batch_z})
        Dsum.append(Dl)
        Gsum.append(Gl)
    if(epoch%10==0):
        print("on epoch",epoch,"    Gloss", Gl,"    Dloss",Dl)
    sample_z=np.random.uniform(-1,1,size=(1,100))
    gen_sample=sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
    samples.append(gen_sample)
    if(epoch%10==0):
        plt.imshow(samples[epoch].reshape(28,28))
        plt.show()
sample_z=np.random.uniform(-1,1,size=(1,100))
gen_sample=sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
plt.imshow(gen_sample.reshape(28,28))

