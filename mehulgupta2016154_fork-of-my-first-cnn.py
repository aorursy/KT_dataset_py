# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x=pd.read_csv('../input/train.csv')

y=pd.read_csv('../input/test.csv')
z=x['label']

x=x.drop(['label'],axis=1)
x=x.applymap(lambda f :float(f/255.0))

y=y.applymap(lambda f :float(f/255.0))

x=pd.DataFrame(x)

y=pd.DataFrame(y)
x=x.values.reshape(-1,28,28,1)

tester=y.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

z=to_categorical(z,num_classes=10)
from sklearn.model_selection import train_test_split as tts

xtrain,xtest,ztrain,ztest=tts(x,z,train_size=0.8)
import tensorflow as tf

def conv2d(x,w):

    weights=tf.Variable(tf.random.normal(w))

    return tf.nn.relu(tf.nn.conv2d(x,weights,[1,1,1,1],padding='SAME'))

def dropout(x,prob):

    return tf.nn.dropout(x,prob)

def max_pool(x,k):

    return tf.nn.max_pool(x,k,[1,2,2,1],padding='SAME')

def fully_connected(x,size):

    weights=tf.Variable(tf.random.normal([int(x.get_shape()[1]),size],mean=0,stddev=1))

    return tf.matmul(x,weights)

def cost_func(x,y):

    cost=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=x)

    cost_1=tf.reduce_mean(cost)

    return cost_1
tf.reset_default_graph()

inp=tf.placeholder(tf.float32,shape=(None,28,28,1),name='input')

target=tf.placeholder(tf.float32,shape=(None,10),name='target')

l1=conv2d(inp,[5,5,1,32])

l2=max_pool(l1,[1,2,2,1])

l9=dropout(l2,0.9)

l3=conv2d(l9,[5,5,32,64])

l4=max_pool(l3,[1,2,2,1])

l0=dropout(l4,0.9)

l5=tf.reshape(l0,[-1,7*7*64])

l6=fully_connected(l5,10)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=l6))

opt=tf.train.AdamOptimizer(0.005).minimize(cost)

prediction=tf.argmax(l6,1)

result=tf.equal(tf.argmax(l6,1),tf.argmax(target,1))

accuracy=tf.reduce_mean(tf.cast(result,tf.float32))

def train_gen():

    image=[]

    label=[]

    count=0

    for x,y in zip(xtrain,ztrain):

                if count<8:

                    count+=1

                    image.append(x)

                    label.append(y)

                if count==8:

                        yield np.array(image).reshape(-1,28,28,1),np.array(label).reshape(-1,10)

                        count=0

                        image=[]

                        label=[]

                

def val_gen():

        image=[]

        label=[]

        count=0

        for x,y in zip(xtest,ztest):

                if count<64:

                    count+=1

                    image.append(x)

                    label.append(y)

                if count==64:

                        yield np.array(image).reshape(-1,28,28,1),np.array(label).reshape(-1,10)

                        count=0

                        image=[]

                        label=[]
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for q in range(5):

        z=train_gen()

        print("EPOCH="+str(q))

        for x,y in z:

                a,b,c=sess.run([cost,opt,accuracy],feed_dict={'input:0':x,'target:0':y})

        result=sess.run([cost,accuracy],feed_dict={'input:0':np.array(xtest).reshape(-1,28,28,1),'target:0':np.array(ztest).reshape(-1,10)})

        print("Epoch "+str(q)+" accuracy"+str(result[1])+" loss="+str(result[0]))

    f=sess.run([prediction],feed_dict={'input:0':tester})

    

    

    

    
f