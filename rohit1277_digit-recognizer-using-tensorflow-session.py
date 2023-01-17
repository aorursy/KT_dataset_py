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
import pandas as pd

from tqdm import tqdm_notebook
train = pd.read_csv("../input/train.csv")
train
y = train["label"]

X = train.drop(labels = ["label"],axis = 1)
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
y =  np.array(y)

X = np.array(X)
index = 100

s= np.reshape(X[index],(28,28))

plt.imshow(s)

print(y[index])
tf.reset_default_graph()

img_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,28,28,1),name="image")

label_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,10),name="label")
def model(img_placeholder):

    w1=tf.get_variable(name="w1",shape=[3,3,1,64],initializer=tf.contrib.layers.xavier_initializer())

    conv1=tf.nn.conv2d(input=img_placeholder,strides=[1,1,1,1],filter=w1,padding="SAME",name="conv1")

    bias1 = tf.get_variable(name="b1",shape=[64],initializer=tf.contrib.layers.xavier_initializer())

    add = tf.nn.bias_add(conv1,bias1)

    act1=tf.nn.leaky_relu(add,name="activation1")

    print(act1)

    w2=tf.get_variable(name="w2",shape=[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())

    conv2=tf.nn.conv2d(input=act1,strides=[1,1,1,1],filter=w2,padding="SAME",name="conv2")

    bias2 = tf.get_variable(name="b2",shape=[128],initializer=tf.contrib.layers.xavier_initializer())

    add2 = tf.nn.bias_add(conv2,bias2)

    act2=tf.nn.leaky_relu(add2,name="activation2")

    print(act2)

    w3=tf.get_variable(name="w3",shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())

    conv3=tf.nn.conv2d(input=act2,strides=[1,1,1,1],filter=w3,padding="SAME",name="conv3")

    bias3 = tf.get_variable(name="b3",shape=[256],initializer=tf.contrib.layers.xavier_initializer())

    add3 = tf.nn.bias_add(conv3,bias3)

    act3=tf.nn.leaky_relu(add3,name="activation3")

    print(act3)

    mpool1 = tf.nn.max_pool(value=act3,ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID",name="maxpooling1")

    print(mpool1)

    w4=tf.get_variable(name="w4",shape=[3,3,256,512],initializer=tf.contrib.layers.xavier_initializer())

    conv4=tf.nn.conv2d(input=mpool1,strides=[1,1,1,1],filter=w4,padding="SAME",name="conv4")

    bias4 = tf.get_variable(name="b4",shape=[512],initializer=tf.contrib.layers.xavier_initializer())

    add4 = tf.nn.bias_add(conv4,bias4)

    act4=tf.nn.leaky_relu(add4,name="activation4")

    mpool2 = tf.nn.max_pool(value=act4,ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID",name="maxpooling2")

    print(mpool2)

    w5=tf.get_variable(name="w5",shape=[3,3,512,1024],initializer=tf.contrib.layers.xavier_initializer())

    conv5=tf.nn.conv2d(input=mpool2,strides=[1,1,1,1],filter=w5,padding="SAME",name="conv5")

    bias5 = tf.get_variable(name="b5",shape=[1024],initializer=tf.contrib.layers.xavier_initializer())

    add5 = tf.nn.bias_add(conv5,bias5)

    act5=tf.nn.leaky_relu(add5,name="activation5")

    print(act5)

    flatten = tf.layers.flatten(act5,name="flatten")

    print(flatten)

    dense1 = tf.layers.dense(flatten,4096,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense1)

    dense2 = tf.layers.dense(dense1,4096,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense2)

    dense3 = tf.layers.dense(dense2,2048,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense3)

    dense4 = tf.layers.dense(dense3,1024,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense4)

    dense5 = tf.layers.dense(dense4,512,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense5)

    dense6 = tf.layers.dense(dense5,512,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense6)

    dense6 =tf.layers.dense(dense6,10,activation=tf.nn.leaky_relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

    print(dense6)

    softmax_out = tf.nn.softmax(logits=dense6)

    return dense6,softmax_out
logits,soft=model(img_placeholder)

print(soft)

loss=tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=label_placeholder),name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

accur = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(soft, 1), tf.argmax(label_placeholder, 1))))

out_label = tf.argmax(soft, axis=1)

valloss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=label_placeholder)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

binencoder = LabelBinarizer()

Y = binencoder.fit_transform(y)
y
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,shuffle=True)
X_train = X_train/784

X_test = X_test/784
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1)
X_train.shape
X_test.shape
epochs = 20

batchsize = 100

val_batchsize = 500
modelpath = "../savedmodel"

if not os.path.exists(modelpath):

    os.mkdir(modelpath)

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    batch_count = X_train.shape[0]//batchsize

    batch_count_v = X_test.shape[0]//val_batchsize

    for epoch in range(epochs):

        for batch in tqdm_notebook(range(batch_count)):

            image_batch = X_train[(batch*batchsize):(batch*batchsize)+batchsize]

            label_batch = y_train[(batch*batchsize):(batch*batchsize)+batchsize]

            _,train_loss, train_accur=sess.run(fetches=[optimizer,loss,accur],feed_dict={img_placeholder:image_batch.astype(np.float32),label_placeholder:label_batch.astype(np.float32)})

        for tbatch in tqdm_notebook(range(batch_count_v)):

            timage_batch = X_test[(tbatch*val_batchsize):(tbatch*val_batchsize)+val_batchsize]

            tlabel_batch = y_test[(tbatch*val_batchsize):(tbatch*val_batchsize)+val_batchsize]

            test_loss,test_accur=sess.run(fetches=[loss,accur],feed_dict={img_placeholder:timage_batch.astype(np.float32),label_placeholder:tlabel_batch.astype(np.float32)})

        print("::epoch number-> ",epoch," :::loss--> ",train_loss," ::testloss ",test_loss," ::trainaccuracy-> ",train_accur," ::test accuracy--> ",test_accur)

    save_path = saver.save(sess,os.path.join(modelpath,"model.ckpt"))

    print("model saved ", save_path)

#         if epoch == 3:

#             predcition1 = sess.run(fetches=[out_label], feed_dict={img_placeholder:np.reshape(X_test[0],(1,28,28,1)).astype(np.float32)})

#             print("+----- ", predcition1)
test = pd.read_csv("../input/test.csv")
test
testdata = np.array(test)
testdata.shape
os.listdir(modelpath)
# tf.reset_default_graph()

# saver = tf.train.Saver()
with tf.Session() as sess:

    saver.restore(sess,os.path.join(modelpath,"model.ckpt"))

    print("Model restored.")

    predcition1 = sess.run(fetches=[out_label], feed_dict={img_placeholder:np.reshape(testdata[8],(1,28,28,1)).astype(np.float32)})

    print("+----- ", predcition1[0][0])

    
sess = tf.Session()

saver.restore(sess,os.path.join(modelpath,"model.ckpt"))

print("Model restored.")
outputfile = pd.read_csv("../input/sample_submission.csv")
outputfile["ImageId"][0]
for e,i in tqdm_notebook(enumerate(range(testdata.shape[0]))):

    predcition = sess.run(fetches=[out_label], feed_dict={img_placeholder:np.reshape(testdata[i],(1,28,28,1)).astype(np.float32)})

#     print("+----- ", predcition[0][0])

    outputfile["ImageId"][e] = e+1

    outputfile["Label"][e] = predcition[0][0]
# index = 100

s= np.reshape(testdata[2],(28,28))

plt.imshow(s)

# print(y[index])
outputfile.to_csv("submission.csv",index=False)
outputfile