# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings;warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pylab as pl,h5py
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
f=h5py.File('../input/classification-of-handwritten-letters/LetterColorImages_123.h5','r')
keys=list(f.keys()); keys
backgd=np.array(f[keys[0]])
lbl=np.array(f[keys[2]])
img=np.array(f[keys[1]])/255
from skimage.color import rgb2gray
img1=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
for i in range(img.shape[0]):
    img1[i,:,:]=rgb2gray(img[i,:,:,:])
plt.imshow(img1[0,:,:])
img1.shape
lbl-=1
lbl=lbl.reshape(lbl.shape[0],1)
backgd=backgd.reshape(backgd.shape[0],1)
from sklearn.preprocessing import OneHotEncoder
encoder1=OneHotEncoder()
encoder2=OneHotEncoder()
ohe_backgd=encoder1.fit_transform(backgd).toarray()
ohe_backgd=ohe_backgd.astype('float32')
ohe_lbl=encoder2.fit_transform(lbl).toarray()
ohe_lbl=ohe_lbl.astype('float32')
targets=np.concatenate((ohe_lbl,ohe_backgd),axis=1)
from sklearn.model_selection import train_test_split
X_train,X_t,y_train,y_t=train_test_split(img1,targets,test_size=0.3,stratify=targets,random_state=42)
X_test,X_valid,y_test,y_valid=train_test_split(X_t,y_t,test_size=0.5,stratify=y_t,random_state=42)
X_train=X_train.reshape([X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
X_valid=X_valid.reshape([X_valid.shape[0],X_valid.shape[1]*X_valid.shape[2]])
width=32
height=32
channels=1
flat=width*height*channels
outlen1=33
outlen2=4
initializer = tf.compat.v1.keras.initializers.he_uniform(seed=42)
print(tf.__version__)
X=tf.placeholder(tf.float32,shape=[None,flat])
y_1=tf.compat.v1.placeholder(tf.float32,shape=[None,len(ohe_lbl[1,:])])
y_2=tf.compat.v1.placeholder(tf.float32,shape=[None,len(ohe_backgd[1,:])])
x_image=tf.reshape(X,[-1,32,32,1])
W_conv1=tf.Variable(initializer(shape=[3,3,1,64]))
b_conv1=tf.Variable(np.zeros([64]),dtype=tf.float32)
W_conv4=tf.Variable(initializer(shape=[3,3,64,64]))
b_conv4=tf.Variable(np.zeros([64]),dtype=tf.float32)
W_conv2=tf.Variable(initializer(shape=[5,5,64,128]))
b_conv2=tf.Variable(np.zeros([128]),dtype=tf.float32)
W_conv3=tf.Variable(initializer(shape=[5,5,128,256]))
b_conv3=tf.Variable(np.zeros([256]),dtype=tf.float32)
convolve1=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1
h_conv1=tf.nn.relu(convolve1)
conv1=tf.compat.v1.layers.batch_normalization(h_conv1)
convolve4=tf.nn.conv2d(conv1,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4
h_conv4=tf.nn.relu(convolve4)
conv4=tf.compat.v1.layers.batch_normalization(h_conv4)
conv4=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
convolve2=tf.nn.conv2d(conv4,W_conv2,strides=[1,1,1,1],padding='VALID')+b_conv2
h_conv2=tf.nn.relu(convolve2)
conv2=tf.compat.v1.layers.batch_normalization(h_conv2)
conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
convolve3=tf.nn.conv2d(conv2,W_conv3,strides=[1,1,1,1],padding='VALID')+b_conv3
h_conv3=tf.nn.relu(convolve3)
conv3=tf.compat.v1.layers.batch_normalization(h_conv3)
conv3.shape
layer2_matrix=tf.reshape(conv3,[-1,256])
layer2_matrix=tf.compat.v1.nn.dropout(layer2_matrix,keep_prob=0.2)
W_fc1=tf.Variable(initializer([256,128]))
b_fc1=tf.Variable(np.zeros([128]),dtype=tf.float32)
W_fc2=tf.Variable(initializer([128,outlen1]))
b_fc2=tf.Variable(np.zeros(outlen1),dtype=tf.float32)
W_fc3=tf.Variable(initializer([128,outlen2]))
b_fc3=tf.Variable(np.zeros([outlen2]),dtype=tf.float32)
fc1=tf.nn.relu(tf.matmul(layer2_matrix,W_fc1)+b_fc1)
fc2=tf.nn.softmax(tf.matmul(fc1,W_fc2)+b_fc2)
fc3=tf.nn.softmax(tf.matmul(fc1,W_fc3)+b_fc3)
y_CNN=[fc2,fc3]
cross_entropy1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1,logits=fc2))
cross_entropy2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2,logits=fc3))
train_step1=tf.train.AdamOptimizer(0.001).minimize(cross_entropy1)
train_step2=tf.train.AdamOptimizer(0.001).minimize(cross_entropy2)
train_step=tf.group(train_step1,train_step2)
total_loss=cross_entropy1+cross_entropy2
correct_predictions=[tf.equal(tf.argmax(fc2,1),tf.argmax(y_1,1)),tf.equal(tf.argmax(fc3,1),tf.argmax(y_2,1))]
accuracy=tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
Epochs=100
def mini_batches(X,y,i,mini_batch_size=64):
    m=int(X.shape[0])
    shuffled_X=tf.random.shuffle(X,seed=i)
    shuffled_y=tf.random.shuffle(y,seed=i)
    num=m//mini_batch_size
    mini_batches_=[]
    for j in range(num):
        mini_batch_X=X[j*mini_batch_size:(j+1)*mini_batch_size,:]
        mini_batch_y=y[j*mini_batch_size:(j+1)*mini_batch_size,:]
        mini_batch= (mini_batch_X,mini_batch_y)
        mini_batches_.append(mini_batch)
    if m%mini_batch_size!=0:
        mini_batch_X=X[num*mini_batch_size:,:]
        mini_batch_y=y[num*mini_batch_size:,:]
        mini_batch= (mini_batch_X,mini_batch_y)
        mini_batches_.append(mini_batch)
    return mini_batches_    
with  tf.compat.v1.Session() as sess:
    tf.compat.v1.disable_eager_execution()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(Epochs):
        mini_batches_1=mini_batches(X_train,y_train,i,256)
        for mini in mini_batches_1:
            (mini_X,mini_y)= mini
            mini_y1=mini_y[:,:33]
            mini_y2=mini_y[:,33:]
            _,loss_val=sess.run([train_step,total_loss],feed_dict={X:mini_X,y_1:mini_y1,y_2:mini_y2})
        print(str(accuracy.eval(feed_dict={X:X_train,y_1:y_train[:,:33],y_2:y_train[:,33:]}))+ "=========="+str(loss_val))
        print(str(accuracy.eval(feed_dict={X:X_valid,y_1:y_valid[:,:33],y_2:y_valid[:,33:]})))
