import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')
labeled_images.shape
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
img_num = labeled_images.shape[0]

def preprocess(data, labeled=True):
    images = []
    if labeled:
        images = data.iloc[:,1:] / 255.
    else:
        images = data / 255.
    
    print(images.shape)
    
    width = height = np.ceil(np.sqrt(images.shape[1])).astype(np.uint8)
    images = np.reshape(np.array(images), (-1, width, height, 1))
    print(images.shape)
    
    labels = []
    if labeled:
        labels = data.iloc[:, :1]
        labels_count = np.unique(labels).shape[0]
        print("There are {} labels.".format(labels_count))
        
        labels = encoder.fit_transform(labels)
        print(labels[2])
        
    return images, labels

images, labels = preprocess(labeled_images, labeled=True)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)
# 
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)
# Three convolutional layers with their channel counts, and a 
# fully connected layer (the last layer has 10 softmax neurons)
# try another value(24,48,64, 200)

K = 6 # first convolutional layer output depth 24
L = 12
M = 24
N = 200

W1 = tf.Variable(tf.truncated_normal([6,6,1,K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5,5,K,L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
W4 = tf.Variable(tf.truncated_normal([7*7*M,N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
# model

s = 1 # means stride
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,s,s,1], padding='SAME') + B1)

s = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,s,s,1], padding="SAME") + B2)

s=2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,s,s,1], padding='SAME') + B3)

YY = tf.reshape(Y3, shape=[-1, 7*7*M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)
# corss entropy loss function, normalized for batchs of 100 images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
train_a = []
test_a = []
train_range = []
test_range = []
batch_size = 100

def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = next_batch(batch_size)
    
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)
    
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1})
        print(a, c, learning_rate)
        
        train_a.append(a)
        train_range.append(i)
    
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={
            X: test_images,
            Y_: test_labels,
            pkeep: 1.0
        })
        print(str(i) + ': ********** epoch ' + str(i*100//train_images.shape[0]+1) + ' ******* test acc: ' + str(a) + ' test loss: ' + str(c))
        
        test_a.append(a)
        test_range.append(i)
        
    sess.run(train_step, feed_dict={
        X: batch_X,
        Y_: batch_Y,
        lr: learning_rate,
        pkeep: 0.75
    })
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global epochs_completed
    global index_in_epoch
    
    
    start = index_in_epoch
    if epochs_completed == 0 and start == 0:
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        train_images = train_images[perm0]
        train_labels = train_labels[perm0]
        
    if start + batch_size > num_examples:
        epochs_completed += 1
        rest_num_examples = num_examples - start
        images_rest_part = train_images[start:num_examples]
        labels_rest_part = train_labels[start:num_examples]
        
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        
        start = 0
        index_in_epoch = batch_size - rest_num_examples
        end = index_in_epoch
        images_new_part = train_images[start:end]
        labels_new_part = train_labels[start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
        index_in_epoch += batch_size
        end = index_in_epoch
        return train_images[start:end], train_labels[start:end]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2000+1):
    training_step(i, i%100==0, i%20==0)
test_ims = pd.read_csv('../input/test.csv')
print(test_ims.shape)

images, _ = preprocess(test_ims, False)
predict = tf.argmax(Y, 1)
test_num = test_ims.shape[0]
predicted_labels = np.zeros(test_num)
batch_size = 100

for i in range(test_num // batch_size):
    start = i*batch_size
    end = (i+1)*batch_size
    predicted_labels[start:end] =  sess.run(predict, feed_dict={
        X: images[start:end],
        pkeep: 0.75
    })

my_submission = pd.DataFrame(
    {'ImageId': range(1, len(test_ims)+1), 'Label': predicted_labels},
)
my_submission.Label = my_submission.Label.astype('int32')
my_submission.to_csv('submission.csv', index=False)
