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
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
X_test = test_df.values
import tensorflow as tf
X = train_df.drop('label',axis=1)
Y = train_df['label']
Y = pd.get_dummies(Y)
X = X.values
Y = Y.values

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None,10])
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100
X.shape[0]
def neural_network_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                     'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                     'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                     'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                     'bias':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.matmul(data , hidden_layer_1['weights']) + hidden_layer_1['bias']
    l1 = tf.nn.relu(l1)
    l2 = tf.matmul(l1 , hidden_layer_2['weights']) + hidden_layer_2['bias']
    l2 = tf.nn.relu(l2)
    l3 = tf.matmul(l2, hidden_layer_3['weights']) + hidden_layer_3['bias']
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3,output_layer['weights']) + output_layer['bias']
    return output
    
def train_neural_network(x):
    prediction = neural_network_model(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction ))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batches = int(X.shape[0]/batch_size)
        for epoch in range(hm_epochs):
            epoch_loss=0
            
            for i in range(total_batches):
                epoch_x = X[i*100:(i+1)*100][:]
                epoch_y = Y[i*100:(i+1)*100][:]
                _, c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch:", epoch , 'completed out of ', hm_epochs,' loss :' ,epoch_loss)
                  

        Pred = sess.run(prediction, feed_dict = {x:X_test})
    return Pred
            
                
                
                
Pred = train_neural_network(x)
type(Pred)
Pred[0:1][:]
A = tf.argmax(Pred,1)
sess = tf.Session()
a = sess.run(A)
a = a.reshape(1,28000)
a.shape
df_output = pd.DataFrame(a)
df_output = df_output.T
df_output = df_output.reset_index()
df_output.columns = ['ImageId','Label']
df_output['ImageId'] = df_output['ImageId']+1
df_output.to_csv('submission.csv', index=False)
