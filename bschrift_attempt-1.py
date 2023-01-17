# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import scipy.misc

import os



def batches(size : int, num : int, dir : str = '../input/train/'):

    fnames = np.random.permutation(os.listdir(dir))

    #print(fnames[0:3])

    i = 0

    for _ in range(num):

        batch = []

        for _ in range(size):

            batch.append(scipy.misc.imread(dir + fnames[i]))

            i += 1

        yield batch

        
class PetClassNN:

    def __init__(self):

        print('...')

        

    def build(self, images, train : bool = False):

        self.x = tf.placeholder(tf.float32)

        in_w, in_h, n_in_c, n_f = images[0].shape

        self.conv_1 = self.conv(self.x, in_w, in_h, n_in_c, n_f, name='conv_1')

        

    def new_weights(self, shape):

        print(shape)

        return tf.Variable(tf.truncated_normal(shape, stddev=0.5))



    def new_biases(self,length):

        return tf.Variable(tf.constant(0.5, shape=[length]))

    

    def conv(self, in_layer, input_width, input_height, num_input_channels, num_filters, name : str, pooling : bool = True):

        weights = self.new_weights(shape=[input_width, input_height, num_input_channels, num_filters])

        biases = self.new_biases(length=num_filters)

        

        conv = tf.conv2d(in_layer,weights,[1,1,1,1],'SAME',True)

        conv += biases

        

        if pooling:

            pool = tf.nn.max_pool(value=conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        

        relu = tf.nn.relu(pool)

        

        return relu



batch = batches(2,2)

PetClassNN().build(next(batch))