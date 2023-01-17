# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
train_data = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
train_data.head()
train_data.shape, test_data.shape
train_data.head()
plt.figure(figsize=(15,10))
sns.countplot(train_data['label'])
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()
encoder = OneHotEncoder() # encoding target variable i.e. label

x_train = (train_data.iloc[:, 1:]/255).values # normalizing train images to standard 0-1 pixel values.
y_train = encoder.fit_transform(train_data['label'].values.reshape(-1,1)).toarray()

x_test = (test_data.iloc[:, 1:]/255).values # normalizing test images to standard 0-1 pixel values.
y_test = encoder.fit_transform(test_data['label'].values.reshape(-1,1)).toarray()
x_train.shape, y_train.shape, x_test.shape, y_test.shape
input_width = 28
input_height = 28
input_channel = 1
input_pixels = 784

n_conv1 = 64
n_conv2 = 128
stride_conv1 = 1
stride_conv2 = 1
filter1_k = 5
filter2_k = 5
maxpool1_k = 2
maxpool2_k = 2

n_hidden = 1024
n_out = 24

input_size_to_hidden_layer = ((input_width//(maxpool1_k*maxpool2_k)) * (input_height//(maxpool1_k*maxpool2_k)) * n_conv2)
weights = {
    'wc1' : tf.Variable(tf.random_normal([filter1_k, filter1_k, input_channel, n_conv1])), # weight corresponding to convolutional layer1.
    'wc2' : tf.Variable(tf.random_normal([filter2_k, filter2_k, n_conv1, n_conv2])), # weight corresponding to convolutional layer2.
    'wh' : tf.Variable(tf.random_normal([input_size_to_hidden_layer, n_hidden])),  # weight corresponding to hidden layer.
    'wo' : tf.Variable(tf.random_normal([n_hidden, n_out])) # weight corresponding to output layer.
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([n_conv1])), # biases corresponding to convolutional layer1.
    'bc2' : tf.Variable(tf.random_normal([n_conv2])), # biases corresponding to convolutional layer2.
    'bh' : tf.Variable(tf.random_normal([n_hidden])), # biases corresponding to hidden layer.
    'bo' : tf.Variable(tf.random_normal([n_out])) # biases corresponding to output layer.
}
# function to get the output from a convolutional layer.
def conv(x, weights, bias, stride = 1):
    output = tf.nn.conv2d(x, weights, padding='SAME', strides=[1, stride, stride, 1])
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output) # applying activation function.
    return output
# function which return output of pooling layer used to decrease image size so that we have to train less weights and biases.
def maxpooling(x, k):
    return tf.nn.max_pool(x, padding='SAME', ksize=[1, k, k, 1], strides=[1, k, k, 1])
def forward_propagation(x, weights, biases):
    x = tf.reshape(x, shape = [-1, input_width, input_height, input_channel])
    
    conv1 = conv(x, weights['wc1'], biases['bc1'], stride_conv1)
    conv1_pool = maxpooling(conv1, maxpool1_k)
    
    conv2 = conv(conv1_pool, weights['wc2'], biases['bc2'], stride_conv2)
    conv2_pool = maxpooling(conv2, maxpool2_k)
    
    hidden_layer_input = tf.reshape(conv2_pool, shape = [-1, input_size_to_hidden_layer])
    hidden_layer_output = tf.nn.relu(tf.add(tf.matmul(hidden_layer_input, weights['wh']), biases['bh']))
    
    output = tf.add(tf.matmul(hidden_layer_output, weights['wo']), biases['bo'])
    return output
X = tf.placeholder(tf.float32, [ None, input_pixels], name='x')
Y = tf.placeholder(tf.int32, [ None, n_out], name='y')
pred = forward_propagation(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))

# using adam optimizer on the cost.
optimizer = tf.train.AdamOptimizer(learning_rate=0.011)
optimize = optimizer.minimize(cost)
# creating a new session of tensorflow.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size=64
a = 0
for i in range(10):
    num_batches = int(len(x_train)/batch_size)
    total_cost = 0
    for j in range(num_batches):
        batch_x = x_train[a: a+batch_size]
        batch_y = y_train[a: a+batch_size]
        c, _ = sess.run([cost, optimize], feed_dict={X:batch_x, Y:batch_y})
        total_cost += c
        a += batch_size
    a = 0
    print('total cost at',i+1,'iteration:',total_cost)
# testing model with training data.
predictions = tf.argmax(pred, axis=1)
correct_labels = tf.argmax(Y, axis=1)
accuracy = tf.equal(predictions, correct_labels)
predictions, labels, accuracy = sess.run([predictions, correct_labels, accuracy], feed_dict={X:x_train, Y:y_train})
accuracy.sum()/len(x_train)
# testing model with testing data.
predictions = tf.argmax(pred, axis=1)
correct_labels = tf.argmax(Y, axis=1)
accuracy = tf.equal(predictions, correct_labels)
predictions, labels, accuracy = sess.run([predictions, correct_labels, accuracy], feed_dict={X:x_test, Y:y_test})
accuracy.sum()/len(x_test)
