import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random
print(os.listdir())
TRAINING_DIR = './training_set/training_set/'
TEST_DIR = './test_set/test_set/'
def get_data(name):
    x = []
    y = []
    for Dir in os.listdir(name):
        if not Dir.startswith('.'):
            if Dir in ['cats']:
                value = 0
            elif Dir in ['dogs']:
                value = 1
                
            for file in os.listdir(name+'/'+Dir):
                img = cv2.imread(name+'/'+Dir+'/'+file) 
                if img is not None:
                    img = cv2.resize(img,(50, 50))
                    img = np.array(img)
                    x.append(img)
                    y.append(value)
    return np.array(x), np.array(y)
trainx, trainy = get_data(TRAINING_DIR)
testx, testy = get_data(TEST_DIR)
trainx.shape
class BatchGenerator():
    where = 0
    
    def __init__(self, x, y, batch_size, one_hot = False, nb_classes = 0):
        self.nb_classes = nb_classes
        self.one_hot = one_hot
        self.x_ = x
        self.y_ = y
        self.batch_size = batch_size
        
        self.total_batch = int(len(x) / batch_size)
        self.x = self.x_[:batch_size,]
        self.y = self.y_[:batch_size,]
        self.where = batch_size
        
        if self.one_hot :
            self.set_one_hot()

    def next_batch(self):
        if self.where + self.batch_size > len(self.x_) :
            left = len(self.x_) - self.where
            self.x = self.x_[len(self.x_) - left:]
            self.y = self.y_[len(self.y_) - left:]
            self.where = 0
            
            if self.one_hot:
                self.set_one_hot()
            
            
        else:
            self.x = self.x_[self.where:self.where+self.batch_size,]
            self.y = self.y_[self.where:self.where+self.batch_size,]
            self.where += self.batch_size
        
            if self.one_hot:
                self.set_one_hot()
        
    def set_one_hot(self):
        one_hot = np.array(self.y).reshape(-1)
        self.y = np.eye(self.nb_classes)[one_hot]
        
def create_weight(size, name):
    return tf.Variable(tf.random_normal(size, dtype=tf.float32, stddev=0.01), name = name)
    
def create_bias(size,name):
    return tf.Variable(tf.random_normal(size, dtype=tf.float32, stddev=0.01), name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pooling(x, size, strides):
    return tf.nn.max_pool(x, ksize=size, strides=strides, padding='SAME')
tf.reset_default_graph()

X = tf.placeholder(shape=[None,50,50,3], dtype=tf.float32)
Y = tf.placeholder(shape=[None,2], dtype=tf.float32)


W1 = create_weight([3,3,3,32],'W1')
b1 = create_bias([32],'b1')

W2 = create_weight([3,3,32,64],'W2')
b2 = create_bias([64],'b2')

W3 = create_weight([3,3,64,128],'W3')
b3 = create_bias([128],'b3')

W4 = create_weight([3,3,128,256],'W4')
b4 = create_bias([256],'b4')

W_out1 = create_weight([50*50*256, 512],'W_out1')
b_out1 = create_bias([512],'b_out1')

W_out2 = create_weight([512,2],'W_out2')
b_out2 = create_bias([2],'b_out2')

def model(x, train = True):
    conv2d_1 = max_pooling(tf.nn.relu(conv2d(X, W1) + b1),[1,4,4,1],[1,1,1,1])
    conv2d_2 = max_pooling(tf.nn.relu(conv2d(conv2d_1, W2) + b2),[1,4,4,1],[1,1,1,1])
    conv2d_3 = max_pooling(tf.nn.relu(conv2d(conv2d_2, W3) + b3),[1,4,4,1],[1,1,1,1])
    conv2d_4 = max_pooling(tf.nn.relu(conv2d(conv2d_3, W4) + b4),[1,4,4,1],[1,1,1,1])

    
    FC1 = tf.reshape(conv2d_4, [-1, 50*50*256])
        
    if train:
        FC1 = tf.nn.relu(tf.matmul(FC1, W_out1) + b_out1)
        FC1 = tf.nn.dropout(FC1, keep_prob=0.7)
    
    else:
        FC1 = tf.nn.relu(tf.matmul(FC1, W_out1) + b_out1)
        
    FC2 = tf.matmul(FC1, W_out2) + b_out2
    
    return FC2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model(X), labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch = BatchGenerator(trainx, trainy, batch_size=50, nb_classes=2, one_hot=True)
epoches = 1000

print('started!')
for epoch in range(epoches):
    avg_cost = 0
    for i in range(batch.total_batch):
        feed_dict = {X: batch.x, Y: batch.y}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / batch.total_batch
        batch.next_batch()
        
    print('Epoch:' '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))        
correct_prediction = tf.equal(tf.argmax(model(X, False), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('----- TRAIN DATA ACCURACY -----')
accu  = 0
batch = BatchGenerator(trainx, trainy, batch_size=30, nb_classes=2,one_hot=True)
for i in range(batch.total_batch):
    feed_dict = {X:batch.x, Y:batch.y}
    accu += sess.run(accuracy, feed_dict=feed_dict)
    batch.next_batch()

print( (accu / batch.total_batch) * 100 , '%' )


print('----- TEST DATA ACCURACY -----')
accu  = 0
test = BatchGenerator(testx, testy, batch_size=100,one_hot=True, nb_classes=2)
for i in range(test.total_batch):
    feed_dict = {X:test.x, Y:test.y}
    accu += sess.run(accuracy, feed_dict=feed_dict)
    test.next_batch()

print( (accu / test.total_batch) * 100 , '%' )
test = BatchGenerator(testx, testy, batch_size=len(testx),one_hot=True, nb_classes=2)

for i in range(9):
    
    r = random.randint(0, len(test.x))
    plt.subplot(3,3,i+1)

    plt.title('Label: {}, Pre: {}'.format(sess.run(tf.argmax(test.y[r:r+1], 1)),
                                                  sess.run(tf.argmax(model(X), 1), 
                                                           feed_dict={X: test.x[r:r+1]})))
    b,g,r = cv2.split(test.x[r])
    plt.imshow( cv2.merge([r,g,b]))
    plt.tight_layout()
