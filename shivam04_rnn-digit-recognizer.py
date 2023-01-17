# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import tensorflow as tf

from tensorflow.contrib import rnn

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
epochs = 30

n_classes = 10

batch_size = 105

chunk_size = 28

n_chunks = 28

rnn_size = 105

x = tf.placeholder('float',[None,n_chunks,chunk_size])

y = tf.placeholder('float')
data = pd.read_csv('../input/train.csv')

print(data.head())
data_set = data.as_matrix()

print(data_set[:5])
X_train = data_set[:,1:]

y_train = data_set[:,0]

print(X_train.shape)

print(y_train.shape)
data_test = pd.read_csv('../input/test.csv')

print(data_test.head())
data_set_test = data_test.as_matrix()

print(data_set_test[:5])
X_test = data_set_test[:,:]

print(X_test.shape)
y_n_train = np.zeros((y_train.size,n_classes)).astype(int)

print(y_n_train.shape)
k = 0

for i in y_train:

    y_n_train[k,i] = 1

    k+=1

print(y_n_train)
X_train = X_train.astype(float)

X_test = X_test.astype(float)

y_n_train = y_n_train.astype(float)

print(type(X_train[0,0]))

print(type(X_test[0,0]))

print(type(y_n_train[0,0]))
def recurrent_neural_network(x):

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),

             'biases':tf.Variable(tf.random_normal([n_classes]))}



    x = tf.unstack(x, n_chunks, 1)

    #print x.shape

    lstm_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)



    # Get lstm cell output

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)



    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']



    return output
def train_neural_network(x):

    prediction = recurrent_neural_network(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    

    

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())



        for epoch in range(epochs):

            epoch_loss = 0

            i = 0

            while i<len(X_train):

                start = i

                end = i + batch_size

                #print(start,end)

                epoch_x, epoch_y = X_train[start:end],y_n_train[start:end]

                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                #print epoch_x.shape

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

                i += batch_size



            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)



        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))



        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:X_train.reshape((-1, n_chunks, chunk_size)), y:y_n_train}))

        pred = sess.run(prediction,feed_dict={x:X_test.reshape((-1, n_chunks, chunk_size))})

        corr = tf.argmax(pred,1)

        corr = sess.run(corr)

        print(corr,len(corr))

        k = [i+1 for i in range(len(corr))]

        yg = pd.DataFrame({'ImageId':pd.Series(k),'Label':pd.Series(corr)})

        yg.to_csv('label.csv',index=False)

train_neural_network(x)