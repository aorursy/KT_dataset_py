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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import tensorflow as tf
observations=1000

xs=np.random.uniform(-10,10,(observations,1))

zs=np.random.uniform(-10,10,(observations,1))

generated_inputs=np.column_stack((xs,zs))

noise=np.random.uniform(-1,1,(observations,1))

generated_targets=2*xs+5*zs-15+noise

np.savez('TF_intro',inputs=generated_inputs,targets=generated_targets)

input_size=2

output_size=1

inputs=tf.placeholder(tf.float32,[None,input_size])

targets=tf.placeholder(tf.float32,[None,output_size])

weights=tf.Variable(tf.random_uniform([input_size,output_size],-0.1,0.1))

biases=tf.Variable(tf.random_uniform([output_size],-0.1,0.1))

outpots=tf.matmul(inputs,weights)+biases
mean_loss=tf.losses.mean_squared_error(labels=targets,predictions=outpots)/2
optimize=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)
sess=tf.InteractiveSession()
intializer=tf.global_variables_initializer()
sess.run(intializer)
training_data=np.load('TF_intro.npz')
for i in range(100):

    _,curr_loss=sess.run([optimize,mean_loss],feed_dict={inputs:training_data['inputs'],targets:training_data['targets']})

    print(curr_loss)
out=sess.run([outpots],feed_dict={inputs:training_data['inputs']})
out
sns.distplot(np.squeeze(out)-np.squeeze(training_data['targets']))