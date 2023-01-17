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
x_train=pd.read_csv('../input/train.csv')
x_test_data=pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split
x1=x_train.iloc[:,1:785]

y1=x_train.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.10,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape,x_test_data.shape
x_train=x_train/255

x_test=x_test/255

x_test_data=x_test_data/255
x_train.shape,x_test.shape,x_test_data.shape
import tensorflow as tf

from keras.utils import to_categorical 
classes=10

y_train=to_categorical(y_train,num_classes = classes)

y_test=to_categorical(y_test,num_classes = classes)
y_train[0] #x_train=tf.to_float(x_train)
epochs=30

batch_size=64

display_progress=40

wt_init=tf.contrib.layers.xavier_initializer()
n_input=784



n_conv_1=32

k_conv_1=3



n_conv_2=64

k_conv_2=3



n_conv_3=128

k_conv_3=3



pool_size=2

mp_dropout=0.25



n_dense=128

dense_dropout=0.5



n_dense2=128



n_classes=10
x=tf.placeholder(tf.float32,[None,n_input])

y=tf.placeholder(tf.float32,[None,n_classes])
def dense(x,W,b):

    z=tf.add(tf.matmul(x,W),b)

    a=tf.nn.relu(z)

    return a



def conv_2d(x,W,b,stride_length=1):

    xw=tf.nn.conv2d(x,W,strides=[1,stride_length,stride_length,1],padding='SAME')

    z=tf.nn.bias_add(xw,b)

    a=tf.nn.relu(z)

    return a



def maxpooling2d(x,p_size):

    return tf.nn.max_pool(x,

                         ksize=[1,p_size,p_size,1],

                         strides=[1,p_size,p_size,1],

                         padding='SAME')
def network(x,weights,biases,n_in,mp_size,mp_dropout,den_dropout):

    

    sqr_dim=int(np.sqrt(n_in))

    sqr_x=tf.reshape(x,shape=[-1,sqr_dim,sqr_dim,1])

    

    conv1=conv_2d(sqr_x,weights['W_c1'],biases['b_c1'])

    conv2=conv_2d(conv1,weights['W_c2'],biases['b_c2'])

    conv3=conv_2d(conv2,weights['W_c3'],biases['b_c3'])

    conv4=conv_2d(conv3,weights['W_c4'],biases['b_c4'])

    pool1=maxpooling2d(conv4,mp_size)

    pool1=tf.nn.dropout(pool1,1-mp_dropout)

    

    conv5=conv_2d(pool1,weights['W_c5'],biases['b_c5'])

    conv6=conv_2d(conv5,weights['W_c6'],biases['b_c6'])

    conv7=conv_2d(conv6,weights['W_c7'],biases['b_c7'])

    conv8=conv_2d(conv7,weights['W_c8'],biases['b_c8'])

    pool2=maxpooling2d(conv8,mp_size)

    pool2=tf.nn.dropout(pool2,1-mp_dropout)

    #conv9=conv_2d(pool2,weights['W_c5'],biases['b_c5'])

    

    #conv9=conv_2d(pool2,weights['W_c9'],biases['b_c9'])

    #conv10=conv_2d(conv9,weights['W_c10'],biases['b_c10'])

    #conv11=conv_2d(conv10,weights['W_c11'],biases['b_c11'])

    #conv12=conv_2d(conv11,weights['W_c12'],biases['b_c12'])

    #pool3=maxpooling2d(conv12,mp_size)

    #pool3=tf.nn.dropout(pool3,1-mp_dropout)

    

    flat = tf.reshape(pool2, [-1, weights['W_d1'].get_shape().as_list()[0]])

    dense_1 = dense(flat, weights['W_d1'], biases['b_d1'])

    dense_2 = dense(dense_1, weights['W_d2'], biases['b_d2'])

    dense_2 = tf.nn.dropout(dense_2, 1-den_dropout)

    

    # output layer: 

    out_layer_z = tf.add(tf.matmul(dense_2, weights['W_out']), biases['b_out'])

    #print(out_layer_z.shape)

    return out_layer_z

    
bias_dict = {

    'b_c1': tf.Variable(tf.zeros([n_conv_1])),

    'b_c2': tf.Variable(tf.zeros([n_conv_1])),

    'b_c3': tf.Variable(tf.zeros([n_conv_1])),

    'b_c4': tf.Variable(tf.zeros([n_conv_1])),

    'b_c5': tf.Variable(tf.zeros([n_conv_2])),

    'b_c6': tf.Variable(tf.zeros([n_conv_2])),

    'b_c7': tf.Variable(tf.zeros([n_conv_2])),

    'b_c8': tf.Variable(tf.zeros([n_conv_2])),

    #'b_c9': tf.Variable(tf.zeros([n_conv_3])),

    #'b_c10': tf.Variable(tf.zeros([n_conv_3])),

    #'b_c11': tf.Variable(tf.zeros([n_conv_3])),

    #'b_c12': tf.Variable(tf.zeros([n_conv_3])),

    'b_d1': tf.Variable(tf.zeros([n_dense])),

    'b_d2': tf.Variable(tf.zeros([n_dense2])),

    'b_out': tf.Variable(tf.zeros([n_classes]))

}



# calculate number of inputs to dense layer: 

full_square_length = np.sqrt(n_input)

pooled_square_length = int(full_square_length / (pool_size*pool_size))

dense_inputs = pooled_square_length**2 * n_conv_2



weight_dict = {

    'W_c1': tf.get_variable('W_c1',  [k_conv_1, k_conv_1, 1, n_conv_1], initializer=wt_init),

    'W_c2': tf.get_variable('W_c2',  [k_conv_1, k_conv_1, n_conv_1, n_conv_1], initializer=wt_init),

    'W_c3': tf.get_variable('W_c3',  [k_conv_1, k_conv_1, n_conv_1, n_conv_1], initializer=wt_init),

    'W_c4': tf.get_variable('W_c4',  [k_conv_1, k_conv_1, n_conv_1, n_conv_1], initializer=wt_init),

    'W_c5': tf.get_variable('W_c5',  [k_conv_2, k_conv_2, n_conv_1, n_conv_2], initializer=wt_init),

    'W_c6': tf.get_variable('W_c6',  [k_conv_2, k_conv_2, n_conv_2, n_conv_2], initializer=wt_init),

    'W_c7': tf.get_variable('W_c7',  [k_conv_2, k_conv_2, n_conv_2, n_conv_2], initializer=wt_init),

    'W_c8': tf.get_variable('W_c8',  [k_conv_2, k_conv_2, n_conv_2, n_conv_2], initializer=wt_init),

    #'W_c9': tf.get_variable('W_c9',  [k_conv_3, k_conv_3, n_conv_2, n_conv_3], initializer=wt_init),

    #'W_c10': tf.get_variable('W_c10',  [k_conv_3, k_conv_3, n_conv_3, n_conv_3], initializer=wt_init),

    #'W_c11': tf.get_variable('W_c11',  [k_conv_3, k_conv_3, n_conv_3, n_conv_3], initializer=wt_init),

    #'W_c12': tf.get_variable('W_c12',  [k_conv_3, k_conv_3, n_conv_3, n_conv_3], initializer=wt_init),

    'W_d1': tf.get_variable('W_d1',  [dense_inputs, n_dense], initializer=wt_init),

    'W_d2': tf.get_variable('W_d2',  [n_dense, n_dense2], initializer=wt_init),

    'W_out': tf.get_variable('W_out',  [n_dense2, n_classes], initializer=wt_init)

}
predictions = network(x, weight_dict, bias_dict, n_input, 

                      pool_size, mp_dropout, dense_dropout)
print(predictions.shape)
print(y.shape)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions ,labels=y))

optimizer=tf.train.AdamOptimizer().minimize(cost)
correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))

accuracy_pct=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100
init=tf.global_variables_initializer()
predict=tf.argmax(predictions,1)
#x_train.shape[0]

#x_train.shape,y_train.shape

#keep_prob = tf.placeholder('float')
#batch_start_idx = (1 * batch_size) % (x_train.shape[0] - batch_size)

#batch_end_idx = batch_start_idx + batch_size

#batch_X = x_train[batch_start_idx:batch_end_idx]

#batch_Y = y_train[batch_start_idx:batch_end_idx]

#batch_X.shape,batch_Y.shape
with tf.Session() as session:

    session.run(init)

    

    print("Training for", epochs, "epochs.")

    

    # loop over epochs: 

    for epoch in range(epochs):

        

        avg_cost = 0.0 # track cost to monitor performance during training

        avg_accuracy_pct = 0.0

        

        # loop over all batches of the epoch:

        n_batches = int(x_train.shape[0] / batch_size)

        #batchnumber=0

        for i in range(n_batches):

            

            # batch_x, batch_y = mnist.train.next_batch(batch_size)

            #batchnumber= batchnumber+1

            batch_start_idx = (i * batch_size) % (x_train.shape[0] - batch_size)

            batch_end_idx = batch_start_idx + batch_size

            batch_X = x_train[batch_start_idx:batch_end_idx]

            batch_Y = y_train[batch_start_idx:batch_end_idx]

            

            # feed batch data to run optimization and fetching cost and accuracy: 

            _, batch_cost, batch_acc, Predict = session.run([optimizer, cost, accuracy_pct, predictions], 

                                                   feed_dict={x: batch_X, y: batch_Y})

            

            # accumulate mean loss and accuracy over epoch: 

            avg_cost += batch_cost / n_batches

            avg_accuracy_pct += batch_acc / n_batches

            

        # output logs at end of each epoch of training:

        print("Epoch ", '%03d' % (epoch+1), 

              ": cost = ", '{:.3f}'.format(avg_cost), 

              ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%", 

              sep='')

    

    print("Training Complete. Testing Model.\n")

    

    test_cost = cost.eval({x: x_test, y: y_test})

    test_accuracy_pct = accuracy_pct.eval({x: x_test, y: y_test})

    

    print("Test Cost:", '{:.3f}'.format(test_cost))

    print("Test Accuracy: ", '{:.2f}'.format(test_accuracy_pct), "%", sep='')

    

    #predicted_lables = predict1.eval({x: x_train})

    #print(len(predicted_lables))

    predicted_lables = np.zeros(x_test_data.shape[0])

    for i in range(0,x_test_data.shape[0]//batch_size):

        predicted_lables[i*batch_size : (i+1)*batch_size] = predict.eval({x: x_test_data[i*batch_size : (i+1)*batch_size], 

                                                                                })

predicted_lables.shape,len(predicted_lables)
np.savetxt('Avinash1.csv', 

                        np.c_[range(1,len(x_test_data)+1),predicted_lables], 

                        delimiter=',', 

                        header = 'ImageId,Label', 

                        comments = '', 

                        fmt='%d')
predicted_lables[0]