# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import random, time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
#print(train_data.columns)
train_label = train_data.loc[:,'label']
train_label = train_label.values
#print(train_label)
train_images = train_data.loc[:,'pixel0':]
train_images = train_images.values
imgsize = int(np.sqrt(train_images.shape[1]))
train_images = train_images.reshape((-1,imgsize,imgsize,1))
print(train_images.shape)

train_img_mean = np.mean(train_images,axis=0)
print(train_img_mean.shape)

train_images = train_images - train_img_mean
test_data = pd.read_csv('../input/test.csv')
#print(test_data.columns)
test_images = test_data.loc[:,:].values.reshape((-1,imgsize,imgsize,1))
#print(test_images.shape)
test_images = test_images - train_img_mean
val_images = train_images[:2000]
#print(val_images.shape)
val_label = train_label[:2000]
train_images = train_images[2000:]
train_label = train_label[2000:]
#print(train_images.shape)
import matplotlib.pyplot as plt
plt.imshow(train_img_mean.reshape(28,28))
plt.show()
import tensorflow as tf

def get_weights(name,shape):
    with tf.variable_scope("weights", reuse=tf.AUTO_REUSE): 
        return tf.get_variable(name=name,shape=shape,initializer = tf.contrib.layers.xavier_initializer(uniform=False),regularizer = tf.contrib.layers.l2_regularizer(tf.constant(0.001, dtype=tf.float32)))
    
def get_bias(name,shape):
    with tf.variable_scope("bias", reuse=tf.AUTO_REUSE):
        return tf.get_variable(name=name,shape=shape,initializer = tf.zeros_initializer())

def conv2d(inp,name,kshape,s,padding):
    with tf.variable_scope(name) as scope:
        kernel = get_weights('weights',shape=kshape)
        conv = tf.nn.conv2d(inp,kernel,[1,s,s,1],padding)
        bias = get_bias('biases',shape=kshape[3])
        preact = tf.nn.bias_add(conv,bias)
        convlayer = tf.nn.relu(preact,name=scope.name)
    return convlayer

def maxpool(inp,name,k,s):
    return tf.nn.max_pool(inp,ksize=[1,k,k,1],strides=[1,s,s,1],padding='SAME',name=name)

def loss(logits,labels):
    labels = tf.reshape(tf.cast(labels,tf.int64),[-1])
    #print labels.get_shape().as_list(),logits.get_shape().as_list()
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    total_loss = tf.add(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),cross_entropy_mean,name='total_loss')
    return total_loss

def top_1_acc(logits,true_labels):
    pred_labels = tf.argmax(logits,1)
    true_labels = tf.cast(true_labels,tf.int64)
    #print pred_labels.get_shape().as_list(),true_labels
    correct_pred = tf.cast(tf.equal(pred_labels, true_labels), tf.float32)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return accuracy

def top_5_acc(logits,true_labels):
    true_labels = tf.cast(true_labels,tf.int64)
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, true_labels, k=5,name='top_5_acc'), tf.float32))
def get_new_size():
    new_size = 28 + random.choice([6,12,0])
    return [new_size,new_size]

def get_random_augmentation_combinations(length):
    out = [False,True]
    return [random.choice(out) for i in range(length)]
index = np.arange(train_images.shape[0])

#........ This part will used to get training data for each epoch during training
num_epochs = 100
batch_size = 50
numiter = 800
ne = 0

feed_images = tf.placeholder(tf.float32,shape=(None,28,28,1))
feed_labels = tf.placeholder(tf.float32,shape=(None,))
lr = tf.placeholder(tf.float32,shape=())
keep_prob = tf.placeholder(tf.float32,shape=())
aug_img = tf.placeholder(tf.float32,shape=(28,28,1))
#Data Augmentation

img_scale_crop = tf.random_crop(tf.image.resize_images(aug_img,get_new_size()),[28,28,1])
#img_rand_flip_lr = tf.image.random_flip_left_right(aug_img)
#img_rand_flip_ud = tf.image.random_flip_up_down(aug_img)
#img_con = tf.image.random_contrast(aug_img,0.01,0.25)
#img_br = tf.image.random_brightness(aug_img,0.25)
img_rot = tf.contrib.image.rotate(aug_img,1.57*(-0.5+random.random()),interpolation='BILINEAR')
with tf.device('/gpu:0'):
    conv1 = conv2d(feed_images,'conv1',[3,3,1,32],1,'SAME')
    conv2 = conv2d(conv1, 'conv2',[3,3,32,32],1,'SAME')
    pool1 = maxpool(conv2,'pool1',2,2)
    #size = N,14,14,32
    conv3 = conv2d(pool1,'conv3',[3,3,32,64],1,'SAME')
    conv4 = conv2d(conv3,'conv4',[3,3,64,64],1,'SAME')
    pool2 = maxpool(conv4,'pool2',2,2)
    #size N,7,7,64
    conv5 = conv2d(pool2,'conv5',[3,3,64,128],1,'VALID')
    #size N,6,6,64
    conv6 = conv2d(conv5,'conv6',[3,3,128,128],1,'SAME')
    #size N,6,6,128
    pool3 = maxpool(conv6,'pool3',2,2)
    #size N,3,3,128
    conv7 = conv2d(pool3,'conv7',[1,1,128,256],1,'SAME')
    #size N,3,3,256
    flatpool = tf.contrib.layers.flatten(conv7)
    fc1 = tf.contrib.layers.fully_connected(flatpool,1024,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.001, dtype=tf.float32)))
    dropout1 = tf.nn.dropout(fc1,keep_prob)
    fc2 = tf.contrib.layers.fully_connected(dropout1,1024,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.001, dtype=tf.float32)))
    dropout2 = tf.nn.dropout(fc2,keep_prob)
    logits = tf.contrib.layers.fully_connected(dropout2,10,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.001, dtype=tf.float32)))
    
    cost = loss(logits,feed_labels)

    opt_mom = tf.train.AdamOptimizer(learning_rate=0.0001)#,momentum=0.9)
    opt = opt_mom.minimize(cost)

    acc = top_1_acc(logits,feed_labels)
#Defined outside gpu0 device since tf.nn.in_top_k is not supported for gpu kernel
valacc = top_5_acc(logits,feed_labels)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
tl=[]
vl=[]
ta=[]
ta5 = []
va=[]
va5 = []
ne=0
while(ne<20):
    stime = time.time()
    print('Epoch::',ne+1,'...')
    
    #Shuffling the Dataset
    if ne != 0:
        np.random.shuffle(index)
        train_images = train_images[index]
        train_label = train_label[index]
    
    for niter in range(numiter):
    
        if (niter+1)%400==0:
            print('iter..',niter+1)
        
        #Getting next Batch
        offset = niter*batch_size
        x_iter, y_iter = np.array(train_images[offset:offset+batch_size,:,:]), np.array(train_label[offset:offset+batch_size])
        
        #Data Augmentation
        for n in range(batch_size):
            args = get_random_augmentation_combinations(1)
            if args[0]:
                x_iter[n] = sess.run(img_rot,feed_dict={aug_img:x_iter[n]})
        
        feed_trdict={feed_images:x_iter,feed_labels:y_iter,keep_prob:0.6}#,lr:0.01
            
        #Train the optimizer
        sess.run(opt,feed_dict=feed_trdict)

    #Calculate accuracy of Training set
    cc = sess.run(cost,feed_dict=feed_trdict)
    tr_acc = sess.run(acc,feed_dict = {feed_images:x_iter,feed_labels:y_iter,keep_prob:1.0})
    top5_tr_acc = sess.run(valacc,feed_dict = {feed_images:x_iter,feed_labels:y_iter,keep_prob:1.0})
    ta.append(tr_acc)
    ta5.append(top5_tr_acc)
    tl.append(cc)
    
    #Calculate accuracy of Validation set
    val_loss = sess.run(cost,feed_dict = {feed_images:val_images,feed_labels:val_label,keep_prob:1.0})
    top5_val_acc = sess.run(valacc,feed_dict = {feed_images:val_images,feed_labels:val_label,keep_prob:1.0})
    top1_val_acc = sess.run(acc,feed_dict = {feed_images:val_images,feed_labels:val_label,keep_prob:1.0})
    va.append(top1_val_acc)
    va5.append(top5_val_acc)
    vl.append(val_loss)
    
    #print 'Epoch..',ne+1,'...'
    print('Training accuracy-> Top-1::',tr_acc*100,'%','Top-5:: ',top5_tr_acc*100,'%',' Training cost::',cc)
    print('Top-1 Validation accuracy::',top1_val_acc*100,'Top-5 Val Accuracy:: ',top5_val_acc*100,'%',' Val loss: ',val_loss)
    print('Time reqd.::',(time.time()-stime)/60,'mins...')

    ne+=1

plt.plot(ta)
plt.plot(va)
plt.show()
plt.plot(ta5)
plt.plot(va5)
plt.show()
plt.plot(tl)
plt.plot(vl)
plt.show()
test_img_preds = tf.nn.softmax(logits)
test_preds = sess.run(test_img_preds,feed_dict={feed_images:test_images,keep_prob:1.0})
test_pred_label = np.argmax(test_preds,axis=1)
print(test_pred_label)

test_data = {'ImageId':[],'Label':[]}
for i in range(1,28001):
    test_data['ImageId'].append(i)
    test_data['Label'].append(test_pred_label[i-1])
#print(test_data)

test_df = pd.DataFrame(test_data,columns=['ImageId','Label'])

test_df.to_csv('sample_submission.csv',sep=',',index=False,header=True)
