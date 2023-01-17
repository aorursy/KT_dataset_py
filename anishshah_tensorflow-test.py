import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
train_labels = train.iloc[:, 0].values
train_images = train.iloc[:, 1:].values
train_images = np.multiply(train_images, 1./255)

test_images = test.values
test_images = np.multiply(test_images, 1./255)
n_input = 784
n_class = 10
n_hidden1 = 256
n_hidden2 = 256
x = tf.placeholder("float", shape=[None, n_input]) 
y = tf.placeholder("float", shape=[None, n_class])
def neural_network(_X, _weights, _biases):
    
    layer1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    
    out = tf.add(tf.matmul(layer2, _weights['out']), _biases['out'])
    
    return out
weights = {
    'w1': tf.Variable(tf.zeros(shape=[n_input, n_hidden1])),
    'w2': tf.Variable(tf.zeros(shape=[n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.zeros(shape=[n_hidden2, n_class]))
}

biases = {
    'b1': tf.Variable(tf.zeros(shape=[n_hidden1])),
    'b2': tf.Variable(tf.zeros(shape=[n_hidden2])),
    'out': tf.Variable(tf.zeros(shape=[n_class]))
}
nn_2_layers = neural_network(x, weights, biases)
cost = tf.nn.softmax_cross_entropy_with_logits(nn_2_layers, y)
train = tf.train.AdamOptimizer(0.001).minimize(cost)
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
train_labels = dense_to_one_hot(train_labels)
init = tf.initialize_all_variables()
correct_prediction = tf.equal(tf.argmax(nn_2_layers,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * 100 < 42000:
        batch_x = train_images[step*100:(step+1)*100]
        batch_y = train_labels[step*100:(step+1)*100]
        cost = sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0:
            print("Iter:", step*100, "Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        step += 1