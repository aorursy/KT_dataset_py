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
import h5py
import numpy as np
import random
import itertools

f = h5py.File('../input/ppidata.hdf5', 'r')
features = f['.']['features']
labels = f['.']['labels']

line_number = f['.']['nrows'][:][0]

lines, chunk = list(range(line_number)), 500

index_chunks = [lines[i:i+chunk] for i in range(0, len(lines), chunk)]

chunk_lentgh = len(index_chunks)

index_chunks = random.sample(index_chunks, chunk_lentgh)

train_chunks = index_chunks[:100]
test_chunks = index_chunks[100:]

print(len(labels[:]))
print(sum(labels[:]))

all_labels = labels[:]

print(sum(all_labels[all_labels == [0,1]]))
def get_train_data_batch():
    ct=0
    for c in train_chunks:
        rows = len(c)
        yield features[c], labels[c]
        ct+=1
        print(ct, end=',')
n_nodes_hl1 = 1002
n_nodes_hl2 = 2323
n_nodes_hl3 = 100

n_classes = 2
batch_size = 200
hm_epochs = 7

train_data_batch_gen = get_train_data_batch()
train_x,train_y = train_data_batch_gen.__next__()
print("intitial training load done\n",len(train_x[0]))
#test_x,test_y  = get_test_data()
print("loading done")

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float', [None, len(train_y[0])])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                     'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    #tf.reset_default_graph()
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            train_data_batch_gen = get_train_data_batch()
            epoch_loss = 0
            i=0
            while i < 99:
                train_x, train_y = train_data_batch_gen.__next__()


                #batch_x = train_x
                batch_x = train_x
                batch_y = train_y

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i+=1



            print('\nEpoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
            merged = list(itertools.chain(*test_chunks))
            merged = sorted(merged)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('\nAccuracy:',accuracy.eval({x:features[merged], y:labels[merged]}))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:features[merged], y:labels[merged]}))

train_neural_network(x)
labels[[1,2]]