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
!pip install tensornets
import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

data=[]

label=[]

images_cat_class=500

images_dog_class=500

count_cat=0

count_dog=0

for file in os.listdir("../input/train/train"):

    if count_cat<images_cat_class or count_dog<images_dog_class:

        image=cv2.imread(os.path.join("../input/train/train",file))

        image=cv2.resize(image,(224,224))

        if file.startswith("cat") and count_cat<images_cat_class:

            label.append([1,0])

            data.append(image)

            count_cat+=1

        elif file.startswith("dog") and count_dog<images_dog_class:

            label.append([0,1])

            data.append(image)

            count_dog+=1

    else:

        break

data=np.array(data)

label=np.array(label)

print(data.shape)

print(label.shape)
import tensornets as nets
inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

outputs = tf.placeholder(tf.float32, shape=[None, 2])
logits = nets.VGG19(inputs, is_training=True, classes=2)

model = tf.identity(logits, name='logits')
loss = tf.losses.softmax_cross_entropy(outputs, logits)

train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
epoch=10

batch_size=10
# from tqdm import tqdm

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     for iterate in range(epoch):

#         batch_number_count=data.shape[0]//batch_size

#         train_loss=0.0

#         train_accuracy=0.0

#         for batch in tqdm(range(batch_number_count)):

#             images_train=data[(batch*batch_size):(batch*batch_size)+batch_size,:,:,:]

#             label_train=label[(batch*batch_size):(batch*batch_size)+batch_size,:]

#             print("image_shape",images_train.shape)

#             print("label_shape",label_train.shape)

#             print(label_train)

#             _,train_loss,train_accuracy=sess.run([train,loss,accuracy],feed_dict={inputs:images_train,outputs:label_train})

#         print("epoch",iterate,"loss",train_loss,"accuracy",train_accuracy)
from tqdm import tqdm

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iterate in range(epoch):

        batch_count_total=data.shape[0]//batch_size

        loss_list=[]

        accuracy_list=[]

        for batch in tqdm(range(batch_count_total)):

            _,train_loss,train_accuracy=sess.run([train,loss,accuracy],feed_dict={inputs:data[(batch*batch_size):(batch*batch_size)+batch_size,:,:,:],outputs:label[(batch*batch_size):(batch*batch_size)+batch_size,:]})

            loss_list.append(train_loss)

            accuracy_list.append(train_accuracy)

        print("epoch",iterate,"loss",sum(loss_list)/batch_count_total,"accuracy",sum(accuracy_list)/batch_count_total)