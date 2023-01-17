import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import tensorflow as tf
import matplotlib.pyplot as plt

#import and cut datas
train_data = pd.read_csv('../input/fashion-mnist_train.csv', dtype='float32')
train_data = np.array(train_data)
train_Y = train_data[:,[0]]
train_X = train_data[:,1:]
# check train images
check = tf.Session()
for r in range(0,10):
    plt.subplot(3,5,r+1)
    print(check.run(tf.arg_max(train_X[r:r+1],1)), end=' ')
    plt.imshow(train_X[r:r+1].reshape(28, 28),
              cmap = 'Greys', interpolation='nearest')
    plt.tight_layout()
check.close()
__author__ = "cheayoung jung <che9992@gmail.com>"
__version__ = "2018-06-01"

class BatchGenerator():
    where = 0
    '''
    usage
    
    var = BatchGenerator(xdata, ydata, batch_size = 100)
    var.x 
    var.y
    var.next()
    
    '''
    
    def __init__(self, x, y, batch_size, one_hot = False, nb_classes = 0):
        self.nb_classes = nb_classes
        self.one_hot = one_hot
        self.x_ = x
        self.y_ = y
        self.batch_size = batch_size
        
        self.total_batch = int(len(x) / batch_size)
        self.x = self.x_[:batch_size,:]
        self.y = self.y_[:batch_size,:]
        self.where = batch_size
        
        if self.one_hot :
            self.set_one_hot()

    def next_batch(self):
        if self.where + self.batch_size > len(self.x_) :
            self.where = 0
            
        self.x = self.x_[self.where:self.where+self.batch_size,:]
        self.y = self.y_[self.where:self.where+self.batch_size,:]
        self.where += self.batch_size
        
        if self.one_hot:
            self.set_one_hot()
        
    def set_one_hot(self):
        self.y = np.int32(self.y)
        one_hot = np.array(self.y).reshape(-1)
        self.y = np.eye(self.nb_classes)[one_hot]
tf.reset_default_graph()

# there are 10 of type fashion items 
nb_classes = 10

# set tf variables 
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

W1 = tf.get_variable('W1',shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable('W2',shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.get_variable('W3',shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable('W4',shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(layer3, W4) + b4
hypothesis = tf.nn.softmax(logits)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))



train = tf.train.GradientDescentOptimizer(learning_rate= 0.03).minimize(cost)

datas = BatchGenerator(train_X, train_Y, batch_size=200, one_hot=True, nb_classes=nb_classes)


#prediction 
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(cost, feed_dict = {X: datas.x, Y: datas.y})
# run!
for step in range(100):
    av_cost = 0
    for i in range(datas.total_batch):
        co_v, _ = sess.run([cost, train], feed_dict = {X: datas.x, Y: datas.y})
        av_cost += co_v / datas.total_batch
        datas.next_batch()
    

    print('step : {} cost: {:.8f}'.format(step, av_cost),end='\t')
    print('Accuracy: {:.9f}% ' .format(
        accuracy.eval(
        session=sess, 
        feed_dict = {X: datas.x, Y: datas.y}) * 100))

import random
#import and cut datas
test_data = pd.read_csv('../input/fashion-mnist_test.csv', dtype='float32')
test_data = np.array(train_data)
test_Y = test_data[:,[0]]
test_X = test_data[:,1:]

test = BatchGenerator(test_X, test_Y, batch_size=50000 ,one_hot=True, nb_classes=10)


print('----- TEST -----')
print('Accuracy: {:.2f}% \n' 
      .format(accuracy.eval(session=sess, feed_dict={X:test.x, Y:test.y}) * 100 ,'%'))

for i in range(9):
    
    r = random.randint(0, 1000 - 1)
    plt.subplot(3,3,i+1)

    plt.title('Label: {}, Pre: {}'.format(sess.run(tf.arg_max(test.y[r:r+1], 1)),
                                                  sess.run(tf.arg_max(hypothesis, 1), 
                                                           feed_dict={X: test.x[r:r+1]})))
    plt.imshow(test.x[r:r+1].reshape(28, 28),
          cmap = 'Greys', interpolation='nearest')
    plt.tight_layout()

