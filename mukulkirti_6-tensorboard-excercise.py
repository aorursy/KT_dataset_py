## Import the library

#this tutorial is on tensorflow 1.4 so disabling version 2.0 and enable v1.4

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

import numpy as np
#You create the data. It is an array of 10000 rows and 5 columns

X_train = (np.random.sample((10000,5)))

y_train =  (np.random.sample((10000,1)))

X_train.shape

feature_columns = [

      tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]

DNN_reg = tf.estimator.DNNRegressor(feature_columns=feature_columns,

      # Indicate where to store the log file    

      model_dir="./",    

      hidden_units=[500, 300],    

      optimizer=tf.train.ProximalAdagradOptimizer(      

      learning_rate=0.1,      

      l1_regularization_strength=0.001    

      )

)

#The last step consists to train the model. During the training, TensorFlow writes information in the model directory.

# Train the estimator

train_input = tf.estimator.inputs.numpy_input_fn(    

     x={"x": X_train},    

     y=y_train, shuffle=False,num_epochs=None)

DNN_reg.train(train_input,steps=3000) 
#view log file generated 

import os

os.listdir('./')
%load_ext tensorboard
"""

if tensorboard is not installed then install it by this command

!pip install tensorflow-tensorboard

"""
#Open Tensorboard

%tensorboard --logdir ./
"""

in kaggle some time tensor board not working so you can download 



the files in /kaggle/workind directory in your local machine 

the go to the command prompt

then follow these steps:

1. put all downloaded file in a folder  , lets path of that folder = c:\dir\

2. if tensorboard not install in your system then install it by

   typing cmd as pip install tensorflow-tensorboard

3. then again open cmd and traverse to the directroy c:\dir\

4. then run tensorboard --logdir logs

5. them open the url given in the instruction

"""



#if output is not showing then you can refer this link

# https://colab.research.google.com/drive/19tOFYiTMzmHfqTdxllyMHm7aQVISkMiK