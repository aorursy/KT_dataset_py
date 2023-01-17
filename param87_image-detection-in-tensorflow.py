# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#numpy for matrix manipulotion

import numpy as np

#pandas for file operations

import pandas as pd

#machine learning library

import tensorflow as tf

#for the viualization

import matplotlib.pyplot as plt
#the following two fuctions help in reaading the data

def read_features_from_csv(filename,usecols = range(1, 785)):

    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)

    features = np.divide(features, 255.0)

    return features



def read_labels_from_csv(filename):

    labels_original = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)

    labels = np.zeros([len(labels_original),10])

    labels[np.arange(len(labels_original)), labels_original] = 1

    labels = labels.astype(np.float32)

    return labels
features = read_features_from_csv("../input/fashion-mnist_train.csv")

labels = read_labels_from_csv('../input/fashion-mnist_train.csv')
plt.imshow(np.reshape(features[10],(28,28)),cmap='gray_r')

plt.show()
labels.shape

#same way you cn use the padas just with 2 line of code to read the data

data = pd.read_csv('../input/fashion-mnist_train.csv', delimiter=',', header = 0)



data = data.iloc[:,1:]#to get image of size 28x28 from cloumn 1-785

data.shape

data = data.values

plt.imshow(np.reshape(data[10],(28,28)),cmap='gray_r')

plt.show()
#reset the tensorflow graph 

tf.reset_default_graph()
#input to the graph

x = tf.placeholder(tf.float32, shape = [None,784])

y_ = tf.placeholder(tf.float32, shape = [None,10])



#reshape the  x to feaature 2d image

x_image = tf.reshape(x, [-1, 28,28, 1])
#convolutional layer 1

w_conv1 = tf.Variable( tf.truncated_normal( [5, 5, 1, 32], stddev=0.1))

b_conv1 = tf.Variable( tf.constant(0.1, shape=[32]))



h_conv1 = tf.nn.relu( tf.nn.conv2d(input = x_image, filter =w_conv1, strides=[ 1, 1, 1, 1], padding="SAME") + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



#convolutional layer 2

w_conv2 = tf.Variable( tf.truncated_normal( [5, 5, 32, 64], stddev=0.1))

b_conv2 = tf.Variable( tf.constant(0.1, shape=[64]))



h_conv2 = tf.nn.relu( tf.nn.conv2d(input = h_pool1, filter = w_conv2, strides=[ 1, 1, 1, 1], padding="SAME") + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding="SAME")





#fully connected layer 1

w_fc1 = tf.Variable( tf.truncated_normal([7 * 7 * 64, 1024],stddev = 0.1))

b_fc1 = tf.Variable( tf.constant(0.1, shape = [1024]))



h_pool2_flat =  tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, w_fc1) + b_fc1)



#dropout

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)





#fully connected layer2

w_fc2 = tf.Variable( tf.truncated_normal( [1024, 10], stddev = 0.1))

b_fc2 = tf.Variable( tf.constant( 0.1, shape = [10]))



y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
cross_entropy  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=y, labels= y_))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction  = tf.equal( tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#hyperparameters which place main role in deciding the correctness / accuracy of your model

BatchSize = 50

TrainSplit = 0.999

TrainigStep = 1000
def generate_batch(features, labels, batch_size):

    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)

    batch_features = features[batch_indexes]

    batch_labels = labels[batch_indexes]

    

    return (batch_features, batch_labels)
#split the data into training and validation

train_samples = int( len(features) / (1 / TrainSplit))



train_features = features[: train_samples]

train_labels   = labels[: train_samples]



validation_features = features[train_samples: ]

validation_labels = labels[train_samples: ]

accuracy_history = []

for i in range(TrainigStep):

    

    batch_features, batch_labels = generate_batch(train_features, train_labels, BatchSize)

    

    if i%100 == 0:

        accuracy_ = sess.run( accuracy, feed_dict = {x : validation_features, y_: validation_labels, keep_prob:1.0})

        accuracy_history.append(accuracy_)

        print("step  %i  and validation acc :%g "%(i, accuracy_))



    sess.run(train_step, feed_dict = { x: batch_features, y_: batch_labels, keep_prob:0.5})
ftest = read_features_from_csv('../input/fashion-mnist_test.csv')

ltest = read_labels_from_csv('../input/fashion-mnist_test.csv')
#accuracy for test data

acc = accuracy.eval(feed_dict={x:ftest, y_:ltest, keep_prob:1.0})



print("acc:",acc)
plt.imshow(np.reshape(ftest[10],(28,28)),cmap='gray_r')

plt.show()
ltest[10]