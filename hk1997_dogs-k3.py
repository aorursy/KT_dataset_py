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
#importing dependecies to create dataset

from PIL import Image

import matplotlib.pyplot as plt

import os

import random
#function takes a file name and returns output label in the form of one hot vector

#[0,1]=dog [1,0]=cat

def output_label(l):

    lb=l.split('.')[0]

    #print(ar)

    if str(lb)==str('cat'):

        return [1,0]

    else:

        return [0,1]

   

#function below loads an image, resizes it and returns corresponding numpy array

def load_image(path,width):

    img=Image.open(path)

    img = img.resize((width,width))  

    a=np.array(img)

    return a
#defining constants to be used to prepare our dataset

TRAINING_PATH='../input/train'

ls_train=os.listdir(TRAINING_PATH)
#let's make our training set now

train_set=[]

i=0

for s in ls_train:

    train_set.append([np.array(load_image(os.path.join(TRAINING_PATH,s),64))

                      ,output_label(s)])

    i=i+1

    if i%1000==0:

        print('images processed so far',i)
import tensorflow as tf
#defining placeholders

X=tf.placeholder(tf.float32,shape=(None,64,64,3))

Y_=tf.placeholder(tf.float32,shape=[None,2]) #one hot vectors
#defining helper functions for conv2d and max_pooling

def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')



def max_pooling(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#defining helper functions for weights and bias

def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)



def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)
w1=weight_variable([5,5,3,32])

b1=bias_variable([32])

#now reshaping images from a image_count*784 vector to image_count*28*28*1

#X_image=tf.reshape(X,[-1,28,28,1])



h_conv1=tf.nn.relu(conv2d(X,w1)+b1)

h_pool1=max_pooling(h_conv1)
#defininf second layer of cnn

w2=weight_variable([5,5,32,64])

b2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,w2)+b2)

h_pool2=max_pooling(h_conv2)
h_pool_flat=tf.reshape(h_pool2,[-1,16*16*64])

w3=weight_variable([16*16*64,1024])

b3=bias_variable([1024])



h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat,w3)+b3)
w4=weight_variable([1024,2])

b4=bias_variable([2])

y_conv= tf.matmul(h_fc1,w4)+b4
cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#creating training and testing sets

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split as tts

train_set = shuffle(train_set)

train, test=tts(train_set,test_size=0.1,random_state=1)
print(len(train),len(test))
train_features=np.array([i[0]/255.0 for i in train])

train_labels=np.array([i[1] for i in train])

print(train_features.shape,train_labels.shape)
print('yo')

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)
print("now entering training")

costs=[]

#training with learning rate 0.0001 for 30 epochs to start with 

for i in range (30):   

    j=0

    batch_cost=0

    for j in range (100):

        batch_x=(train_features[(j*225):(j+1)*225])

        batch_y=(train_labels[j*225:(j+1)*225])

        _,cost_batch=sess.run([train_step,cross_entropy],{X:batch_x,Y_:batch_y})

        batch_cost+=cost_batch/100.0   

        if j==99 and j!=0:

            costs.append(batch_cost)

            print(batch_cost)

        if j%49==0:

            print('on minibatch iteration ',j)

    #_x.append(i)  

    #costs.append(float(cost))

    #if i%5==0:

    print("iteration= ",i)

    #print(cost,costs)

plt.plot(costs)

plt.show()


test_features=np.array([i[0]/255 for i in test])

test_labels=np.array([i[1] for i in test])

print('starting to predict test accuracy')

print(sess.run(accuracy,{X:test_features,Y_:test_labels}))
#now trying to predict results for real test set

#loading testing images

TEST_PATH='../input/test'

ls_test=sorted(os.listdir(TEST_PATH))

print(len(ls_test))
test_dict={}

for i in ls_test:

    s=i.split('.')[0]

    test_dict[s]=i
#test_features

test_features=[]

for i in range(1,12501):

    test_features.append(load_image(os.path.join(TEST_PATH,test_dict[str(i)]),64))

    if i%1000==0:

        print('features loaded successfully',i)
test_features=np.array(test_features)

print(test_features.shape)
ans=tf.argmax(y_conv, 1)

test_result=[]

for i in range(125):

    f1=test_features[i*100:(i+1)*100]/255

    pred=sess.run(ans,{X:f1})

    for j in range(len(pred)):

        test_result.append(pred[j])

print(len(test_result))
test_result=np.array(test_result)

labels=[]

for x in range (1,12501):

    labels.append(x)

labels=np.array(labels)
df={

    'id':labels,

    'label':test_result

}

print(len(labels),len(test_result))

pd_df=pd.DataFrame(data=df)

pd_df.to_csv('cnn_regularized2.csv',index=False)