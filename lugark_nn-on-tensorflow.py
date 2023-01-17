import tensorflow as tf



import pandas as pd

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# data = pd.read_csv("D:/ucheba/machine learning/kaggle/digits/train.csv").values.astype("float32")

test = pd.read_csv("D:/ucheba/machine learning/kaggle/digits/test.csv").values.astype("float32")

X = mnist.train.images

y = mnist.train.labels



Xvalid = mnist.validation.images

yvalid = mnist.validation.labels



Xtest = mnist.test.images

ytest = mnist.test.labels



graph = tf.get_default_graph()

sess = tf.InteractiveSession()



XX = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



w1_initial = np.random.normal(size=(784, 100)).astype(np.float32)

w2_initial = np.random.normal(size=(100, 100)).astype(np.float32)

w3_initial = np.random.normal(size=(100, 10)).astype(np.float32)

eps = 1e-3



w_1 = tf.Variable(w1_initial)

bias_1 = tf.Variable(tf.zeros([100]))

w_2 = tf.Variable(w2_initial)

bias_2 = tf.Variable(tf.zeros([100]))

w_3 = tf.Variable(w3_initial)

bias_3 = tf.Variable(tf.zeros([10]))



# ---------- Layer 1 ------

z_1 = tf.matmul(XX, w_1)

b_mean1, b_var1 = tf.nn.moments(z_1, [0])

b_norm1 = (z_1 - b_mean1) / tf.sqrt(b_var1 + eps)

scale1 = tf.Variable(tf.ones([100]))

BN1 = scale1 * b_norm1 + bias_1

p_1 = tf.sigmoid(BN1)



# ---------- Layer 2 ------

z_2 = tf.matmul(p_1, w_2)

b_mean2, b_var2 = tf.nn.moments(z_2, [0])

b_norm2 = (z_2 - b_mean2) / tf.sqrt(b_var2 + eps)

scale2 = tf.Variable(tf.ones([100]))

BN2 = tf.nn.batch_normalization(z_2, b_mean2, b_var2, bias_2, scale2, eps)

p_2 = tf.nn.sigmoid(BN2)



# ---------- Layer 3 ------

z_3 = tf.matmul(p_2, w_3)

b_mean3, b_var3 = tf.nn.moments(z_3, [0])

b_norm3 = (z_3 - b_mean3) / tf.sqrt(b_var3 + eps)

scale3 = tf.Variable(tf.ones([10]))

p_3 = tf.nn.batch_normalization(z_3, b_mean3, b_var3, bias_3, scale3, eps)

prediction = tf.nn.softmax(p_3)



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(p_2) + (1. - y_) * tf.log(1. - p_2), reduction_indices=[1]))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_3, labels=y_))



tr_step = tf.train.AdamOptimizer(0.004).minimize(cross_entropy)



init = tf.global_variables_initializer()



sess.run(init)

for it in range(50000):

    batch_xs, batch_ys = mnist.train.next_batch(50)

    sess.run(tr_step, feed_dict={XX: batch_xs, y_: batch_ys})

    if it % 100 == 0:

        print(it, sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_3, labels=y_)),

                           feed_dict={XX: batch_xs, y_: batch_ys}), sep=' cross_entropy = ')



tf.summary.FileWriter("D:/ucheba/machine learning/kaggle", sess.graph)



corr_preds = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(corr_preds, tf.float32))

print(sess.run(accuracy, feed_dict={XX: X, y_: y}))

print(sess.run(accuracy, feed_dict={XX: Xtest, y_: ytest}))



predictions = sess.run(tf.argmax(prediction, 1), feed_dict={XX: test})



submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("D:/ucheba/machine learning/kaggle/subm.csv", index=False, header=True)
