import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print('train.columns',train.columns)

print('train.shape',train.shape)

print('test.columns',test.columns)

print('test.shape',test.shape )
train_image = train.iloc[:,1:].values

train_label = train.iloc[:,0].values

test_image = test.values

y_train = np.zeros((len(train_label),10))

for i in range(len(train_label)):

    y_train[i][train_label[i]] = 1

train_label = y_train
def batch(np1,np2,b_size):

    size = len(np1)

    random_m = np.random.randint(low=0,high=size-1,size=[b_size])

    return np1[random_m],np2[random_m]        
filter_size = 5

num_filter1 = 32

num_filter2 = 64

max_pool_size = 2

img_size = 28

img_shape = [28,28]

img_size_flat = 28*28

img_channel = 1

num_fc = 1024

num_class = 10

batch_size = 200

LEARNING_RATE_BASE = (1e-7)

LEARNING_RATE_DECAY = 1-(1e-5)

MOVING_AVERAGE_DECAY = 0.99
def get_weights(shape):

    return tf.Variable(tf.random_normal(shape))

def get_biases(length):

    return tf.Variable(tf.random_normal([length]))



def conv_layer(d_input,

              num_input_channels,

              filter_size,

              num_filters,):

    shape = [filter_size,filter_size,num_input_channels,num_filters]

    weights = get_weights(shape)

    biases = get_biases(num_filters)

    layer_1 = tf.nn.conv2d(d_input,

                       filter=weights,

                       strides=[1,1,1,1],

                       padding='SAME' )

    layer_2 = layer_1 + biases

    layer_3 = tf.nn.max_pool(value=layer_2,

                        ksize=[1,max_pool_size,max_pool_size,1],

                        strides = [1,max_pool_size,max_pool_size,1],

                        padding='SAME')

    layer = tf.nn.relu(layer_3)

    return layer,weights



def flatten_layer(layer):

    layer_shape = layer.shape

    n = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer,[-1,n])

    return layer_flat,n



def fc_layer(input,

            num_inputs,

            num_outputs,):

    weights = get_weights(shape = [num_inputs,num_outputs])

    biases = get_biases(length = num_outputs )

    layer_fc = (tf.matmul(input,weights) + biases)

    return layer_fc
def forward(input):  

    x_img = tf.reshape(input,[-1,img_size,img_size,img_channel])

    layer1,w1 = conv_layer(x_img,

                        img_channel,

                        filter_size,

                        num_filter1)

    layer2,w2 = conv_layer(layer1,

                        num_filter1,

                        filter_size,

                        num_filter2)

    layer_flat,n_feature = flatten_layer(layer2)  

    fc1 = fc_layer(layer_flat,

                n_feature,

                num_fc) 

    fc1 = tf.nn.relu(fc1)

    fc2 = fc_layer(fc1,

                num_fc,

                num_class)  

    return fc2
def backward():

    x = tf.placeholder(tf.float32,shape=[None,img_size_flat])

    y_labels = tf.placeholder(tf.float32,[None,num_class])

    y_output = forward(x)

    #y = tf.nn.softmax(y_output)

    global_step = tf.Variable(0,trainable=False)

    LEARNING_RATE = tf.train.exponential_decay(

            LEARNING_RATE_BASE,

            global_step,

            len(train_image)/batch_size,

            LEARNING_RATE_DECAY,

            staircase = True)  

    a1 = tf.equal(tf.argmax(y_output,1),tf.argmax(y_labels, 1))

    acc1 = tf.reduce_sum(tf.cast(a1, tf.float32))

    ap = tf.reduce_mean(tf.reduce_max(y_output,1)/tf.reduce_sum(y_output,1))

    test_output = tf.argmax(y_output,1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_output,labels=y_labels)

    loss = tf.reduce_mean(cross_entropy) 

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)

    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(loss)

    #train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)



    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step,ema_op]):  

        train_op = tf.no_op(name="train")



    fd_test = {x:train_image,y_labels:train_label}

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()       

        sess.run(init_op)                

        STEP =10000        

        for i in range(STEP):   

            images,labels = batch(train_image,train_label,batch_size)

            fd = {

              x: images,

              y_labels: labels

            }

            _,loss_ = sess.run([train_step,loss],feed_dict=fd)            

            acc_t1 = sess.run(acc1,feed_dict=fd_test)                  

            if  i % 1000 == 0 :

                print('Train step:{0}     \tLoss:{1}   \t Accuracy:{2:.2%}  ({3}/42000)    '

                      .format(i,loss_,acc_t1/42000,acc_t1))

        out = sess.run(test_output,feed_dict = {x:test_image,y_labels:np.zeros([28000,10])})

        ap_ = sess.run(ap,feed_dict = {x:test_image,y_labels:np.zeros([28000,10])})

    return out,ap_

test_result,average_p = backward()
print(average_p)
test_result = pd.Series(test_result,name="Label")

test_result = pd.concat([pd.Series(range(1,28001),name = "ImageId"),test_result],axis = 1)

test_result.to_csv("mnist_submission.csv",index=False)

test_result.head()