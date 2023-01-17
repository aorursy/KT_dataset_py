import numpy as np
import pandas as pd
import tensorflow as tf

import time
from matplotlib import pyplot as plt
# model parameters/variables
learning_rate =  0.0003
batch_size = 128
n_epochs = 30

# Placeholder for feature and target
X = tf.placeholder(tf.float32, [batch_size, 784], name = 'features')
y = tf.placeholder(tf.float32, [batch_size, 10], name = 'target')

# initialize weight and bias
W = tf.Variable(tf.random_normal(shape = [784, 10], stddev = 0.01), name = 'weights')
b = tf.Variable(tf.zeros([1,10]), name = 'bias')
# read data
mnist = pd.read_csv("../input/train.csv", dtype = np.float32)

#divide data into test and train
mnist = mnist.reindex(np.random.permutation(mnist.index))

mnist_train = mnist.head(30000)
mnist_test = mnist.tail(12000)
mnist_test.head()
def next_batch(dataset, batch_size=None):
    """Extracts labels and features and creates batch"""
    
    # generate samples
    if batch_size:
        dataset = dataset.sample(batch_size)
    
    labels = dataset.iloc[:,0]
    # create one hot encoding for target variable
    labels = pd.get_dummies(labels)

    features = dataset.iloc[:,1:]
    # Scale the data to [0, 1] by dividing out the max value, 255.
    features = features/255

    return features, labels
# show random example
def show_random_example(dataset):
    rand_example = np.random.choice(mnist.index)
    
    _, ax = plt.subplots()
    ax.matshow(dataset.loc[rand_example][1:].values.reshape(28, 28))
    ax.set_title("Label: %i" % dataset.loc[rand_example][0])
    ax.grid(False)
    
show_random_example(mnist)
# Model to optimize
logits = tf.matmul(X, W) + b

#loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits) 
loss = tf.reduce_mean(entropy)

# optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# Train Model
init = tf.global_variables_initializer()
errors = []

with tf.Session() as sess:

    sess.run(init)
    n_batches = int(mnist.shape[0]/batch_size)
    start_time = time.time()
    
    # train for n epoches
    for i in range(n_epochs):
        total_loss = 0
        
        for j in range(n_batches):
            X_batches, y_batches = next_batch(mnist_train, batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={X:X_batches, y:y_batches})
            total_loss += l
            errors.append(loss)
        
        W_out, b_out = sess.run([W, b])
        print ("Average loss at epoch {0} is {1}".format(i, total_loss/n_batches))
    
    print ("Total time {0} sec".format(time.time() - start_time))
    print ("optimization over!!")