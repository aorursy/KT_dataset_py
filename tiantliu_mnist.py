# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/digit-recognizer"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
%matplotlib inline

np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.info())
VALIDATION_SIZE = 2000
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500
DROPOUT = 0.5
BATCH_SIZE = 50
y_train = train['label']
y_train = pd.get_dummies(y_train).values
labels = y_train.astype(np.uint8)
print(y_train.shape[0])
print(y_train[10])
images = train.iloc[:,1:].values.astype(np.float)
images = np.multiply(images,1.0/255.0)
print('images({0[0]},{0[1]})'.format(images.shape))
image_size = images.shape[1]
print(image_size)
labels_flat = train.iloc[:,0].values.ravel()
print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(10,labels_flat[10]))
labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))
validation_images = images[:VALIDATION_SIZE]
validation_labels = y_train[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = y_train[VALIDATION_SIZE:]
print(train_images.shape)
print(validation_images.shape)
def weight_variable(shape):
    Weights = tf.Variable(tf.truncated_normal(shape,stddev = 0.1))
    return Weights

def biases_variable(shape):
    biases = tf.Variable(tf.constant(0.1,shape=shape))
    return biases

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 第一层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])

x = tf.placeholder('float',shape = [None,784])
y_ = tf.placeholder('float',shape = [None,10])

# (4000,28,28,1)
image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(image,W_conv1) + b_conv1)
print(h_conv1.get_shape())  # 4000,28,28,32
h_pool1 = max_pool_2x2(h_conv1) # 4000,14,14,32
print(h_pool1.get_shape())

layer1 = tf.reshape(h_conv1,(-1,28,28,4,8))
layer1 = tf.transpose(layer1,(0,3,1,4,2))
layer1 = tf.reshape(layer1,(-1,28*4,28*8))

W_conv2 = weight_variable([5,5,32,64])
b_cov2 = biases_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_cov2)
print(h_conv2.get_shape())  # 4000,14,14,64
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2.get_shape()) # 4000,7,7,64

layer2 = tf.reshape(h_conv2,(-1,14,14,4,16))
layer2 = tf.reshape(layer2,(0,3,1,4,2))
layer2 = tf.reshape(layer2,(-1,14*4,14*16))

W_fc1 = weight_variable([7 * 7 * 64,1024])
b_fc1 = biases_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
print(h_fc1.get_shape())

keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = biases_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)
#cost function
cross_entroy = - tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entroy)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

predict = tf.argmax(y,1)
epochs_completed = 0
index_in_epoch = 0
number_examples = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > number_examples:
        #finished epoch
        epochs_completed += 1
        #shuffle data
        perm = np.arange(number_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        #start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= number_examples
    end = index_in_epoch
    return train_images[start:end],train_labels[start:end]
print("DONE")
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

train_accuracies = []
validation_accuracies = []
x_range = []

display_step = 1

for i in range(TRAINING_ITERATIONS):
    batch_xs,batch_ys = next_batch(BATCH_SIZE)
    
    if i % display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_xs,
            y_:batch_ys,
            keep_prob:1.0
        })
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={
                x:validation_images[0:BATCH_SIZE],
                y_:validation_labels[0:BATCH_SIZE],
                keep_prob:1.0
            })
            print("train_accuracy / validation_accuracy => %.2f / %.2f for step %d"%(train_accuracy,validation_accuracy,i))
            validation_accuracies.append(validation_accuracy)
        else:
            print("train_accuracy => %.4f for step %d"%(train_accuracy,i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        if i%(display_step) == 0 and i:
            display_step *= 10
    sess.run(train_step,feed_dict={
        x:batch_xs,
        y_:batch_ys,
        keep_prob:DROPOUT
    })

if (VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict = {
        x:validation_images,
        y_:validation_labels,
        keep_prob:1.0
    })
    print("Validation_accuracy => %.4f"%validation_accuracy)
    plt.plot(x_range,train_accuracies,'-b',label = 'Training')
    plt.plot(x_range,validation_accuracies,'-g',label = 'Validation')
    plt.legend(loc = "lower right",frameon = False)
    plt.ylim(ymax = 1.1)
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.show()
test_images = pd.read_csv('../input/digit-recognizer/test.csv').values
test_images = test_images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))
# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))
# output test image and prediction
display(test_images[10])
print ('predicted_lables[{0}] => {1}'.format(10,predicted_lables[10]))

# save results
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
