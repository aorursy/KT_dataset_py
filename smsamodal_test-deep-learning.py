from scipy.io import loadmat

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





mat_train = loadmat('../input/cars-annos/cars_train_annos.mat')

mat_test = loadmat('../input/cars-annos/cars_test_annos.mat')

meta = loadmat('../input/cars-annos2/cars_meta.mat')



labels = list()

for l in meta['class_names'][0]:

    labels.append(l[0])

    

train = list()

for example in mat_train['annotations'][0]:

    label = labels[example[-2][0][0]-1]

    image = example[-1][0]

    train.append((image,label))

    

test = list()

for example in mat_test['annotations'][0]:

    image = example[-1][0]

    test.append(image)



validation_size = int(len(train) * 0.10)

test_size = int(len(train) * 0.20)



validation = train[:validation_size].copy() #800

np.random.shuffle(validation)

train = train[validation_size:]



test = train[:test_size].copy() #1600

np.random.shuffle(test)

train = train[test_size:]
import cv2

import numpy as np

im_list_train=[]

label_list_train=[]

for pic in train:

    image=cv2.imread("../input/stanford-cars-dataset/cars_train/cars_train/"+pic[0])

    if image is not None:

        res = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

        res=np.array(res.astype(float))

        im=np.array(res/255)

        im_list_train.append(im)

        label_list_train.append(pic[1])

    else:

        print("emptytrain")

im_list_train=np.array(im_list_train)

label_list_train=np.array(label_list_train)



im_list_test=[]

label_list_test=[]

for pic in test:

    image=cv2.imread("../input/stanford-cars-dataset/cars_test/cars_test/"+pic[0])

    if image is not None:

        res = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

        res=np.array(res.astype(float))

        im=np.array(res/255)

        im_list_test.append(im)

        label_list_test.append(pic[1])

    else:

        print("emptytest")

im_list_test=np.array(im_list_test)

label_list_test=np.array(label_list_test)



encoded=np.identity(196)
y=[]

for i in range(196):

    y.append([encoded[i],labels[i]])

    

encoded_list_train=[]

for labels in label_list_train:

    for encode in y:

        if labels==encode[1]:

            encoded_list_train.append(encode[0])

encoded_list_train=np.array(encoded_list_train)



encoded_list_test=[]

for labels in label_list_test:

    for encode in y:

        if labels==encode[1]:

            encoded_list_test.append(encode[0])

encoded_list_test=np.array(encoded_list_test)
import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None,100,100,3])

y_true = tf.placeholder(tf.float32,shape=[None,196])
hold_prob = tf.placeholder(tf.float32)
def init_weights(shape):

    init_random_dist = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(init_random_dist)



def init_bias(shape):

    init_bias_vals = tf.constant(0.1, shape=shape)

    return tf.Variable(init_bias_vals)



def conv2d(x, W):

    # x--> [batch,h,w,channels]

    # w---> [filter H, filter w, channels in, channels out]

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2by2(x):

    # x--> [batch,h,w,channels]

    # ksize---> size of the input window

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')



def convolutional_layer(input_x, shape):

    # shape---> [height of the filter, width of the filter, in color channel, output features]

    W = init_weights(shape)

    b = init_bias([shape[3]])

    return tf.nn.relu(conv2d(input_x, W) + b)



def normal_full_layer(input_layer, size):

    input_size = int(input_layer.get_shape()[1])

    W = init_weights([input_size, size])

    b = init_bias([size])

    return tf.matmul(input_layer, W) + b



def next_batch(im_list,encoded_list,step,batch_size):

    step=np.random.randint(5602)

    x=im_list_train[step:step+batch_size,:]

    y=encoded_list_train[step:step+batch_size,:]

    

    return [x,y,step]
# [height of the filter, width of the filter, in color channel, output features]

convo_1 = convolutional_layer(x,shape=[4,4,3,32])

convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])

convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[100,25*25*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,196)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    step=0

    for i in range(500):

        batch = next_batch(im_list_train,encoded_list_train,step,100)

        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        

        # PRINT OUT A MESSAGE EVERY 100 STEPS

        if i%10 == 0:

            

            print('Currently on step {}'+ str(i))

            print('Accuracy is:')

            # Test the Train Model

            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))



            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            

            test_batch=next_batch(im_list_test,encoded_list_test,0,100)

            print(sess.run(acc,feed_dict={x:test_batch[0],y_true:test_batch[1],hold_prob:1.0}))

            print('\n')

        step=batch[2]
print(batch[1])
from sklearn.metrics import classification_report