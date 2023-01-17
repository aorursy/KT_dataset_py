# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.columns
data.info()
def create_label():
    label = []
    for i in data['Man of the Match']:
        if i == 'Yes':
            label.append([1,0])
        elif i == 'No':
            label.append([0,1])
        else:
            print('Wrong input')
    return label
def create_dataset():
    dataset = []
    for i in range(len(data)):
        dataset.append([data['Goal Scored'][i] , data['Ball Possession %'][i],data['Attempts'][i],
                       data['Blocked'][i],data['Saves'][i],data['Pass Accuracy %'][i],
                       data['Passes'][i],data['On-Target'][i],data['Off-Target'][i]])
    return dataset
learning_rate = 0.04
batch_size = 25
num_steps = 501
num_input = 9
num_class = 2

layer1 = 25
layer2 = 40
layer3 = 15

data_set = create_dataset()
label_set = create_label()
train = int(len(data_set)*0.70)
train_dataset = data_set[:train]
train_label = label_set[:train]

test_dataset = data_set[train:]
test_label = label_set[train:]
tf_x = tf.placeholder(tf.float32)
tf_y = tf.placeholder(tf.float32)
#to use weight and biases in model 
weight = {'w1':tf.Variable(tf.truncated_normal([num_input , layer1])) , 
          'w2':tf.Variable(tf.truncated_normal([layer1 , layer2])),
          'w3':tf.Variable(tf.truncated_normal([layer2 , layer3])),
          'out':tf.Variable(tf.truncated_normal([layer3 , num_class]))}
biases = {'b1':tf.Variable(tf.truncated_normal([layer1])) ,
         'b2':tf.Variable(tf.truncated_normal([layer2])),
         'b3':tf.Variable(tf.truncated_normal([layer3])),
         'out':tf.Variable(tf.truncated_normal([num_class]))}
def model(input_data , W , b):
    l1 = tf.nn.relu(tf.matmul(input_data , W['w1']) + b['b1'])
    l2 = tf.nn.relu(tf.matmul(l1 , W['w2']) + b['b2'])
    l3 = tf.nn.relu(tf.matmul(l2 , W['w3']) + b['b3'])    
    output = tf.matmul(l3 , W['out']) + b['out']
    return output
logits = model(tf_x , weight , biases)
pred = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y , logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred , 1) , tf.argmax(tf_y , 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1,num_steps):
        offset = (step * batch_size) % (len(train_dataset) - batch_size)
        feed_dict = {tf_x : train_dataset[offset:(offset+batch_size)] , 
                     tf_y : train_label[offset:(offset+batch_size)]}
        sess.run(optimizer,feed_dict=feed_dict)
        if step % 100 == 0:
            acc , loss = sess.run([accuracy,cost] , feed_dict=feed_dict)
            print('Step : {} \nTrain Accuracy : {:.2f} \nTrain Loss : {:.2f}'.format(step,acc,loss))
        
    print('-----Optimization Finished-----')
        
    print('Test Accuracy : {:.2f}'.format(sess.run(accuracy , feed_dict={tf_x : test_dataset ,
                                                                        tf_y : test_label})))   





