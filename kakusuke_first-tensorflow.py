%matplotlib inline
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
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)
init = tf.initialize_all_variables()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
sess.run(init)
import numpy as np

steps = []
n_epoc = 10
for epoc in range(n_epoc):
  train.reindex(np.random.permutation(train.index))
  images = train.drop('label', 1) * 1.0 / 255.0
  labels = pd.get_dummies(train['label'])
  for i in range(380):
    index = (images.index >= i * 100) & (images.index < (i + 1) * 100)
    batch_xs = images.loc[index].values
    batch_ys = labels.loc[index].values
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
  steps.append([epoc, 'training',  sess.run(accuracy, feed_dict={x: images[:38000], y_: labels[:38000]})])
  steps.append([epoc, 'validation', sess.run(accuracy, feed_dict={x: images[38000:], y_: labels[38000:]})])
steps = pd.DataFrame(steps, columns=['epoc', 'type', 'accuracy'])
%matplotlib inline
import seaborn as sns
sns.lmplot(x='epoc', y='accuracy', hue='type', data=steps)
predict = tf.argmax(y,1)
predication = pd.DataFrame(sess.run(predict, feed_dict={x: test * 1.0 / 255.0}), columns=['Label'], index=(test.index + 1))
predication.to_csv('output.csv', index=True, index_label='ImageId')
