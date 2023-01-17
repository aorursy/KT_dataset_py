# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df=pd.read_csv(os.path.join('../input','train.csv'))
# df.head()
dft=pd.read_csv(os.path.join('../input','test.csv'))
# dft.head()
labels=df['label'].values
X_train=df.drop('label',axis=1).values.astype(np.float32)
X_train/=225
import matplotlib.cm as cm
index=115
image=X_train[index]
image=image.reshape([28,28])
plt.imshow(image,cmap=cm.binary)
plt.axis('OFF')
from sklearn.preprocessing import LabelEncoder
def OneHotEncoding(labels,num_classes):
    
    """To convert each label to their corresponding One hot encoded data"""
    #label_encoder = LabelEncoder()
    #integer_encoded = label_encoder.fit_transform(labels)
    #print(integer_encoded)
    # binary encode
    #onehot_encoder = OneHotEncoder(sparse=False)
    #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #labels_encode=np.zeros((labels.shape[0],num_classes))
    #labels_encode.flat[np.arange(labels.shape[0])*num_classes+labels.ravel()]=1
    #labels_encode=enc.transform(labels)
    #print(onehot_encoded)
    # invert first example
    #inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    #print(inverted)
    c=0
    labels_encoder=np.zeros((labels.shape[0],num_classes))
    for ii in labels:
        #print(ii.dtype)
        labels_encoder[c][ii]=1
        c=c+1
        
    
    return labels_encoder
labels_encode=OneHotEncoding(labels,10)
Y_train=labels_encode.astype(np.uint8)
def weight_initialize(shape):
    return(tf.Variable(tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)))
    
def add_conv2D(inputs,W,stride,padding):
    return tf.nn.conv2d(input=inputs,filter=W,strides=stride,padding=padding)
def add_Max_Pool(inputs,ksize,padding,strides):
    return tf.nn.max_pool(value=inputs,ksize=ksize,padding=padding,strides=strides)
X=tf.placeholder(shape=[None,784],dtype=tf.float32)
Y=tf.placeholder(shape=[None,10],dtype=tf.float32)

def model():
    W1=weight_initialize([5,5,1,32])
    X_=tf.reshape(X,[-1,28,28,1])
    layer1=tf.nn.relu(add_conv2D(X_,W1,[1,1,1,1],'SAME'))#[-1,28,28,32]
    W2=weight_initialize([5,5,32,32])
    layer2=tf.nn.relu(add_conv2D(layer1,W2,[1,1,1,1],'SAME'))
    layer3=add_Max_Pool(layer2,[1,2,2,1],'SAME',[1,2,2,1])#[-1,14,14,32]
    W3=weight_initialize([3,3,32,64])
    layer4=tf.nn.relu(add_conv2D(layer3,W3,[1,1,1,1],'SAME'))#[-1,14,14,64]
    layer5=add_Max_Pool(layer4,[1,2,2,1],'SAME',[1,2,2,1])#[-1,7,7,64]=[num_examples,x,y,nc]
#     layer5=add_fc(layer4,256,tf.nn.relu)
#     layer6=add_fc(layer5,10,tf.nn.softmax)
    layer5=tf.reshape(layer5,[-1,7*7*64])
    W6=weight_initialize([7*7*64,1024])
    layer6=tf.nn.relu(tf.matmul(layer5,W6))
    W7=weight_initialize([1024,10])
    layer7=tf.nn.softmax(tf.matmul(layer6,W7))
    return layer7
    
index=0
start=index
total=X_train.shape[0]
def next_batch(batch_size):
    global index
    global start
    global total
    global X_train
    global Y_train
    index+=batch_size
    if (index>total):
        prm=np.arange(total)
        np.random.shuffle(prm)
        X_train=X_train[prm]
        Y_train=Y_train[prm]
        start=0
        index=batch_size
        epochs+=1
    assert (batch_size<=total)
    return X_train[start:index],Y_train[start:index]
y_=model()
#print(tf.argmax(y_,1).shape)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_))
optimizer=tf.train.AdamOptimizer(0.0001).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), tf.argmax(y_,1)), tf.float32))

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)
def train():
    train_acc=[]
    val_acc=[]
    x_range=[]
    display_acc=10
    num_epochs=100
    for i in range(num_epochs):
        X_batch,Y_batch=next_batch(100)
        if (i%display_acc==0):
            train_acc=accuracy.eval(feed_dict={X:X_batch, Y:Y_batch})
            print(train_acc,i)
        sess.run(optimizer,feed_dict={X:X_batch,Y:Y_batch})    
train()