# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import string
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import _pickle as pickle
import os.path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from skimage import img_as_float

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
csv_file_path = '../input/handwritten_data_785.csv'
def normalize(image_data, a=0.00, b=1.00):
    return a + (image_data - image_data.min()) * (b - a) / (image_data.max() - image_data.min())

def read_data(file_path):
    labels = list()
    images = list()
    
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            labels.append(row[0])
            image = np.array(row[1:], dtype=np.float32)
            image = normalize(image)
            images.append(image)
        
        return np.array(images, dtype=np.float32) , np.array(labels, dtype=np.float32)
X, Y = read_data(csv_file_path)
    
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))
one_hot_enc = OneHotEncoder(sparse=False)
y_one_hot = one_hot_enc.fit_transform(Y.reshape(-1, 1))
print('y_one_hot shape: {}'.format(y_one_hot.shape))
digit_to_letter_map = { k: v for k, v in enumerate(string.ascii_uppercase, 0)}
n_classes = len(digit_to_letter_map)
n_features = X.shape[1]
print(digit_to_letter_map)
print()
print('Features: {}'.format(n_features))
print('Classes: {}'.format(n_classes))
X_train, XX, y_train, yy = train_test_split(X, y_one_hot, test_size=0.4)
X_valid, X_test, y_valid, y_test = train_test_split(XX, yy, test_size=0.6, shuffle=True)
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_valid shape: {}'.format(X_valid.shape))
print('y_valid shape: {}'.format(y_valid.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))
def plot_class_distribution(dataset, title):
    plt.hist(np.argmax(dataset, axis=1))
    plt.title(title)
    plt.show()

plot_class_distribution(y_train, 'Training classes distribution')
plot_class_distribution(y_valid,'Validation classes distribution')
plot_class_distribution(y_test, 'Test classes distribution')
def plot_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

n_samples = 5
for i in range(n_samples):
    index = random.randint(0, X_train.shape[0])
    image = X_train[index].reshape((28, 28))
    title = digit_to_letter_map[np.argmax(y_train[index])]
    plot_image(image, title)
def layer(x, weight_shape, bias_shape, scope, activation):
    weight_stddev = (2.0 / weight_shape[0]) ** 0.5
    weight_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    with tf.variable_scope(scope) as scope:
        W = tf.get_variable('W', weight_shape, initializer=weight_init)
        b = tf.get_variable('b', bias_shape, initializer=bias_init)
        output = tf.matmul(x, W) + b
        if activation == 'softmax':
            return tf.nn.softmax(output)
        if activation == 'relu':
            return tf.nn.relu(output)
        
        return output
def network(x):
    out1 = layer(x, [n_features, 128], [128], scope='layer1', activation='relu')
    out2 = layer(out1, [128, 128], [128], scope='layer2', activation='relu')
    output = layer(out2, [128, n_classes], [n_classes], scope='output', activation='softmax')
    return output

def training(logits, labels, lr):
    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    return optimizer.minimize(loss_op)

def evaluate(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
learning_rate = 0.001
training_epochs = 100
batch_size = 256
display_step = 10
features = tf.placeholder(tf.float32, [None, n_features])
labels = tf.placeholder(tf.float64, [None, n_classes])

logits = network(features)
eval_op = evaluate(logits, labels)
train_op = training(logits, labels, lr=learning_rate)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(training_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            batch_x, batch_y = X_train[i:i+batch_size], y_train[i:i+batch_size]
            feed_dict = {features: batch_x, labels: batch_y}
            sess.run(train_op, feed_dict=feed_dict)
            
        if epoch % display_step == 0:
            valid_feed_dict = {features: X_valid, labels: y_valid}
            train_feed_dict = {features: X_train, labels: y_train}
            train_acc = sess.run(eval_op, feed_dict=train_feed_dict)
            valid_acc = sess.run(eval_op, feed_dict=valid_feed_dict)
            print('Epoch: {0}: Training accuracy: {1} Validation accuracy: {2}'.format(epoch, train_acc, valid_acc))
            
    print('Training complete')
    
    test_feed_dict = {features: X_test, labels: y_test}
    test_acc = sess.run(eval_op, feed_dict=test_feed_dict)
    print('Test accuracy: {}'.format(test_acc))