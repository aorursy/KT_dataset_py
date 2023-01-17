# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import numpy as np
import pandas as pd

data_train = pd.read_csv("../input/fashion-mnist_train.csv")
data_test = pd.read_csv("../input/fashion-mnist_test.csv")
data_train.head()
tf.reset_default_graph()

#set the hyperparameter
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.001
n_epochs = 50
batch_size = 2000
display_step = 10

#Build a calculation graph
#dense->bn->act

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", use_bias = None)
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2", use_bias = None)
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs",
                                   use_bias = None)
logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                      momentum=0.9)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)#labels允许的数据类型有int32, int64
    loss = tf.reduce_mean(xentropy,name="loss")
    
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1) #取值最高的一位
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) #结果boolean转为0,1
#tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for i in range(data_train.shape[0]// batch_size):
            X_batch = data_train.iloc[i*batch_size:(i+1)*batch_size, 1:]
            y_batch = data_train.iloc[i*batch_size:(i+1)*batch_size, 0]
            sess.run(train_op, feed_dict={
                training: True, X: X_batch, y: y_batch
            })
        if (epoch+1) % display_step == 0:
            accuracy_val = sess.run(accuracy, feed_dict={
            X: data_test.iloc[:,1:],
            y: data_test.iloc[:, 0]
            })
            accuracy_train = sess.run(accuracy, feed_dict={
                X: data_train.iloc[:,1:],
                y: data_train.iloc[:, 0]
            })
            print(epoch+1, "Test accuracy:{:.2%}".format(accuracy_val), end = ' ')
            print("Train accuracy:{:.2%}".format(accuracy_train))
    saver.save(sess,"./mnist_bn.ckpt")
    print("Saved")