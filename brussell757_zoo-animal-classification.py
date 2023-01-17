# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
zoo_data = pd.read_csv('../input/zoo.csv')
class_data = pd.read_csv('../input/class.csv')
zoo_data.head()
zoo_data.info()
class_data.head()
class_data.info()
classes = dict(zip(class_data.Class_Number, class_data.Class_Type))
features = np.array(zoo_data.iloc[:,1:-1])
labels = np.array(zoo_data.iloc[:, -1])

labels = tf.one_hot(labels, depth = len(classes))
with tf.Session() as sess:
    labels = labels.eval(session = sess)
    
print(features.shape)
print(labels.shape)
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, (n_x, None), name = 'X')
    Y = tf.placeholder(tf.float32, (n_y, None), name = 'Y')
    
    return X, Y
def initialize_weights(layer_dims):
    seed = 42
    
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), shape = (layer_dims[l], layer_dims[l - 1]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed))
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), shape = (layer_dims[l], 1), dtype = tf.float32, initializer = tf.zeros_initializer())
    return parameters
def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A = X    
    for l in range(1, L):
        Z = tf.matmul(parameters['W' + str(l)], A) + parameters['b' + str(l)]
        A = tf.nn.relu(Z)
    
    Z = tf.matmul(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    return Z
def compute_cost(Y_PRED, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(Y_PRED), labels = tf.transpose(Y)))
    return cost
def build_optimizer(learning_rate, cost):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return optimizer
def build_model(train_data, test_data, layer_dims, learning_rate = 0.01, epoch = 50, batch_size = 32, print_cost = True):
    tf.reset_default_graph()
    
    (n_x, m) = train_data['data'].T.shape
    n_y = train_data['labels'].T.shape[0]
      
    costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_weights(layer_dims)
    Y_PRED = forward_propagation(X, parameters)
    cost = compute_cost(Y_PRED, Y)
    optimizer = build_optimizer(learning_rate, cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(epochs):
            epoch_cost = 0.
            num_batches = m // batch_size
            
            for i in range(num_batches):
                offset = (i * batch_size) % (train_data['labels'].shape[0] - batch_size)
                batch_X = train_data['data'][offset:(offset + batch_size), :]
                batch_Y = train_data['labels'][offset:(offset + batch_size), :]
                _, batch_cost = sess.run([optimizer, cost], feed_dict = {X:batch_X.T, Y:batch_Y.T})
                epoch_cost += batch_cost / num_batches
            
            if m % batch_size != 0:
                batch_X = train_data['data'][num_batches * batch_size:m, :]
                batch_Y = train_data['labels'][num_batches * batch_size:m, :]
                _, batch_cost = sess.run([optimizer, cost], feed_dict = {X:batch_X.T, Y:batch_Y.T})
                epoch_cost += batch_cost / num_batches
                
            if epoch % 50 == 0 and print_cost == True:
                print('Cost after epoch {}: {}'.format(epoch, epoch_cost))
            
            if print_cost == True:
                costs.append(epoch_cost)
                
        plt.figure(figsize = (16,5))
        plt.plot(np.squeeze(costs), c = 'b')
        plt.xlim(0, epochs - 1)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title('Learning Rate: {}'.format(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        print('Parameters have been trained')
        
        predictions = {'classes': tf.argmax(Y_PRED, axis = 0).eval(feed_dict = {X:test_data['data'].T, Y:test_data['labels'].T}),
                       'probabilities': tf.nn.softmax(Y_PRED).eval(feed_dict = {X:test_data['data'].T, Y:test_data['labels'].T})}
        
        correct_preds = tf.equal(tf.argmax(Y_PRED), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, 'float'))
        
        print('Train Accuracy: ', accuracy.eval({X:train_data['data'].T, Y:train_data['labels'].T}))
        print('Test Accuracy: ', accuracy.eval({X:test_data['data'].T, Y:test_data['labels'].T}))
        
    return parameters, predictions
train_data = {'data':train_features, 'labels':train_labels}
test_data = {'data':test_features, 'labels':test_labels}
layer_dims = [train_data['data'].shape[1], 8, len(classes)]
learning_rate = 0.001
epochs = 500
batch_size = 8

parameters, predictions = build_model(train_data, test_data, layer_dims, learning_rate, epochs, batch_size)
import itertools

def plot_confusion_matrix(cm, target_names, title = 'Confusion Matrix', cmap = None, normalize = True):
    accuracy = np.trace(cm)/ float(np.sum(cm))
    misclass = 1 - accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize = (8,6))
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation = 45)
        plt.yticks(tick_marks, target_names)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment = 'center', color = 'white' if cm[i,j] > thresh else 'black')
        else:
            plt.text(j, i, "{:,}".format(cm[i,j]), horizontalalignment = 'center', color = 'white' if cm[i,j] > thresh else 'black')
            
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\naccuracy={:0.4f}; misclass = {:0.4f}'.format(accuracy, misclass))
    
    plt.show()
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(test_data['labels'], axis = 1), predictions['classes'])
plot_confusion_matrix(cm, normalize = False, target_names = classes.values())
