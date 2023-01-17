#Import
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
np.random.seed(1239)
def sigmoid(x):
    return 1/(1+np.exp(-x))
#First we generate the test data

#The synthetic question:
synthetic_questions = np.arange(-1.9, 3.1, 1)
synthetic_students = np.arange(0,2,0.1)
synthetic_logits = synthetic_students.reshape(-1,1) - synthetic_questions.reshape(1,-1)
synthetic_probs = sigmoid(synthetic_logits)
synthetic_data = (synthetic_probs > np.random.rand(synthetic_probs.shape[0],synthetic_probs.shape[1])).astype('float')

synthetic_data
data_shape = synthetic_data.shape
learning_rate = 0.1
tf.reset_default_graph()
X = tf.placeholder(dtype='float' ,shape=data_shape, name="X")
alpha = tf.Variable(initial_value=np.zeros((data_shape[0],1)), name="alpha", dtype='float')
delta = tf.Variable(initial_value=np.zeros((1,data_shape[1])), name="delta", dtype='float')
log_likelihood = tf.reduce_sum(X * tf.log(tf.sigmoid(alpha-delta)) + (1-X) * tf.log(1-tf.sigmoid(alpha-delta)))
cost = -log_likelihood
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(cost)
init = tf.global_variables_initializer()
n_epochs = 4000


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 1000 == 0:
            print("Epoch", epoch, "Cost =", cost.eval(feed_dict={X: synthetic_data}))
        sess.run(training_op, feed_dict={X: synthetic_data})
    
    best_alpha = alpha.eval()
    best_delta = delta.eval()
best_alpha
best_delta