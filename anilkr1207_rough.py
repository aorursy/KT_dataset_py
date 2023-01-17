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

import keras

import random

import numpy as n

import pandas as pd

from scipy.sparse import csr_matrix

from multiprocessing import Pool,Process

import time

import math

from keras import layers

class tmpmodel(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def __call__(self,inputs):

        return self.l(inputs)

def forward(args):

    inputs=args[0]

    m=args[1]

    return m(inputs)

tmpm=tmpmodel()

global fin

fin=tf.zeros((3,300),dtype=tf.float64)

class tatten(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def compute_output_shape(self,input_shape):

        return (3,300)

    def call(self,y):

        atty=self.l(crep)

        atty=tf.cast(atty,dtype=tf.float64)

        tfinr=tf.zeros((3,300),dtype=tf.float64)

        y=tf.cast(y,dtype=tf.float64)

        for i in range(y.shape[0]):

            a=tf.gather(atty,i,axis=0)

            b=tf.gather(y,i,axis=0)

            x=tf.tensordot(a,b,axes=1)

            x=tf.cast(x,dtype=tf.float64)

            x=tf.math.sigmoid(x)

            updates=b*x

            updates=tf.expand_dims(updates,axis=0)

            tfinr=tf.tensor_scatter_nd_add(tfinr,tf.constant([[i]],dtype=tf.int32),updates)

        return tfinr

tmpatt=tatten()

global crep

crep=tf.zeros((3,300),dtype=tf.float64)

for epochs in range(10):

    with tf.GradientTape(persistent=True) as Tape:

        y=forward([tf.convert_to_tensor([[1],[2],[3]],dtype=tf.float32),tmpm])

        crep=tf.zeros((3,300),dtype=tf.float64)

        for i in range(3):

            tmp=tf.ones((1,300),dtype=tf.float64)

            crep=tf.tensor_scatter_nd_add(crep,tf.constant([[i]],dtype=tf.int32),tmp)

            tmp=tf.gather(y,i,axis=0)

            tmp=tf.expand_dims(tmp,axis=0)

            tmp=tf.cast(tmp,dtype=tf.float64)

            crep=tf.tensor_scatter_nd_add(crep,tf.constant([[i]],dtype=tf.int32),tmp)

        crep=crep/3

        #Attention

        crep=tf.cast(crep,dtype=tf.float32)

        fin=tmpatt(y)

        #Error calculation for attention layer

        ans=tf.convert_to_tensor(0,dtype=tf.float64)

        for i in range(y.shape[0]):

            for j in range(y.shape[0]):

                if i!=j:

                    a=tf.gather(fin,i,axis=0)

                    b=tf.gather(fin,j,axis=0)

                    tmp=tf.tensordot(a,b,axes=1)

                    tmp=tf.math.sigmoid(tmp)

                    tmp=-1*tf.math.log(tmp)

                    ans=tf.math.add(ans,tmp)

        att_loss=tf.convert_to_tensor(ans)

        print(att_loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    grads=Tape.gradient(att_loss,tmpatt.trainable_weights)

    optimizer.apply_gradients(zip(grads,tmpatt.trainable_weights))

    grads=Tape.gradient(att_loss,tmpm.trainable_weights)

    optimizer.apply_gradients(zip(grads,tmpm.trainable_weights))