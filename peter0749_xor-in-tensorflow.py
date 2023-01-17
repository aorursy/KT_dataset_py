%matplotlib inline
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
tf.reset_default_graph()

with tf.Graph().as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 1])
    
    W1 = tf.Variable(tf.random_normal([2, 32]))
    b1 = tf.Variable(tf.zeros([32]))
    
    W2 = tf.Variable(tf.random_normal([32, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    
    l1 = tf.nn.tanh((x@W1)+b1)
    l2 = tf.nn.sigmoid((l1@W2)+b2)
    
    loss = -tf.reduce_mean( y*tf.log(l2) + (1-y)*tf.log(1-l2) )
    
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    op = optimizer.minimize(loss)
    
    l2_rounded = tf.round(l2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, l2_rounded), 'float16'))
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
label = (data[:,0]^data[:,1])[...,np.newaxis]

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    loss_history = []
    acc_history = []
    for i in tqdm_notebook(range(3000)):
        l, _, acc = sess.run([loss, op, accuracy], feed_dict={x:data, y:label})
        loss_history.append(l)
        acc_history.append(acc)
    print(sess.run([l2_rounded, accuracy], feed_dict={x:data, y:label}))
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(list(range(len(loss_history))), loss_history, label="binary cross entropy")
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.legend()
ax_score.plot(list(range(len(acc_history))), acc_history, label="accuracy")
ax_score.set_xlabel('epoch')
ax_score.set_ylabel('accuracy')
ax_score.legend()
plt.tight_layout()
plt.show()
