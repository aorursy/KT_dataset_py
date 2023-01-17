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
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

tf.set_random_seed(777)
tf.set_random_seed(777)

class DataFeeder():

    def __init__(self,path):

        data = pd.read_csv(path)
        data_x = data[data.columns.values[:-1]].values##get 20 features
        data_y = data[data.columns.values[-1]].values##get 1 labels
         
        ##Standard Scaler
        ##more efficient than min-max normalization in this data
        scaler = StandardScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)

        data_y_label = set(data_y)
        self.y_dict = {data_y_label.pop():0}

        for step in range(len(data_y_label)):
            self.y_dict.update({data_y_label.pop():len(self.y_dict)})

        data_onehot = np.zeros((len(data_y),len(self.y_dict)),np.int32)
        
        for ix,value in enumerate(data_y):
            idx = self.y_dict[value]
            data_onehot[ix][idx] = 1
       
        data_y = data_onehot

        data_len = len(data_y)
        idx_rand = np.random.choice(data_len, data_len,replace=False)
        idx_train, idx_test, _ = np.split(idx_rand,
                                        [int(data_len*0.8),
                                        data_len])    

        self.x_train = data_x[idx_train]
        self.y_train = data_y[idx_train]
        self.x_test = data_x[idx_test]
        self.y_test = data_y[idx_test]

    def get_nextbatch(self,batch_size=100):

        idx_len = len(self.y_train)
        idx = np.random.choice(idx_len,idx_len, replace=False)

        for step in range(len(idx)//batch_size):
            ix = idx[step*batch_size:(step+1)*batch_size]
            yield self.x_train[ix],self.y_train[ix]



data = DataFeeder('../input/voice.csv') 
bank_size=[20]
num_filters=30
x_features=20
y_labels=2
dense_batch_dims=32
dense_fc1_dims=12
dense_fc2_dims=8
train_epoch=500
print_every_acc=25
learning_rate=0.005
val_acc=[]

tf.reset_default_graph() 
with tf.Session() as sess:

    input_x = tf.placeholder(tf.float32, shape=[None,x_features],name='x')
    input_y = tf.placeholder(tf.int32, shape=[None,y_labels],name='y')
    phase = tf.placeholder(tf.bool,name="phase")

    result=[] 

    ex_x = tf.expand_dims(input_x,1)
    ex_x = tf.expand_dims(ex_x,3)

    ##convolution bank. I user only one size(20)
    ##I make more sence in num_filters(30) than conv bank size(20)
    for i, filter_size in enumerate(bank_size):
        with tf.variable_scope('conv_%s'%i):

            filter_shape = [1,filter_size,1,num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='w_%s'%i)
            b = tf.Variable(tf.constant(0.0,shape=[num_filters]),name='b_%s'%i)
            conv =tf.nn.conv2d(ex_x,W,strides=[1,1,1,1],padding='VALID')

            h = tf.nn.sigmoid(tf.nn.bias_add(conv,b))
            pooled = tf.nn.max_pool(
                h,
                ksize=[1,1,20-filter_size+1,1],
                strides=[1,1,1,1],
                padding='VALID')
            
        result.append(pooled)
        
    result = tf.concat(result,2)
    cnn_output = tf.squeeze(result,1)
    cnn_output = tf.reshape(cnn_output, [-1,num_filters*len(bank_size)])

    def dense_batch_relu(x,phase,scope): 
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(x,dense_batch_dims)
            h2 = tf.contrib.layers.batch_norm(h1,center=True,scale=True,
                                              is_training=phase)
            return tf.nn.relu(h2)

    def dense(x,size,scope):
        return tf.contrib.layers.fully_connected(x,size,
                                                 scope=scope)
    
    ##do batch normalization one time
    l1 = dense_batch_relu(cnn_output,phase,'dense_batch')

    l2 = tf.nn.relu(dense(l1,dense_fc1_dims,'dense_fc_1'),'d2')

    l3 = tf.nn.relu(dense(l2,dense_fc2_dims,'dense_fc_2'),'de')

    logits = dense(l3,y_labels,'logits')

    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=input_y))
    
    ##add loss for batch norma
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(train_epoch):

        for train_x, train_y in data.get_nextbatch():
            feed_dict = {input_x:train_x,input_y:train_y,phase:True}
            t1 = sess.run([train_step], feed_dict=feed_dict)

        if step%print_every_acc==0:        
            feed_dict = {input_x:data.x_train,input_y:data.y_train,phase:True}
            loss1,acc1 = sess.run([loss,accuracy],feed_dict=feed_dict)

            feed_dict = {input_x:data.x_test,input_y:data.y_test,phase:False}
            loss2,acc2 = sess.run([loss,accuracy],feed_dict=feed_dict)
            
            print('[%s]loss = %f , train acc = %f , test acc = %f'%(step,loss1,acc1,acc2))
            val_acc.append(acc2)
            
    print('[Finished]max test accuracy : %f'%max(val_acc))
         







