# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import numpy as np
import tensorflow as tf
import pandas as pd

# Clean up in case repeatedly running in jupyter notebook
tf.reset_default_graph()
# Get reproducable results by making the weight initialization always the same.
tf.set_random_seed(0)

df = pd.read_csv('../input/diabetes.csv')

actualY = df['Outcome'].values
actualX = df.drop(['Outcome'], axis=1).values
actualX = np.array(np.reshape(actualX,newshape=[768,8]))
actualY = np.array(np.reshape(actualY,newshape=[768,1]))
x = tf.placeholder(dtype=tf.float32,shape=[768,8])
W1 = tf.Variable((tf.random_normal(shape=[8,1])))
B1 = tf.Variable(dtype=tf.float32,initial_value=tf.zeros(shape=[1,1]))
y_prediction = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), B1))
y_true = tf.placeholder(dtype=tf.float32, shape=[768,1])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_prediction,labels=y_true))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Compute accuracy
accuracy_op, update_op = tf.metrics.accuracy(labels=y_true, predictions=y_prediction)

with  tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Needed to initialzie accuracy ops
    tf.local_variables_initializer().run()

    for i in range(1000):
        # Compute accuracy every 100th epoch
        if (i + 1) % 100 == 0:
            _, _, accuracy = sess.run([optimizer, update_op, accuracy_op],feed_dict={x: actualX, y_true: actualY})
            print('Epoch: {}, accuracy: {:.3}'.format(i + 1, accuracy))
        else:
            sess.run(optimizer,feed_dict={x: actualX, y_true: actualY})

