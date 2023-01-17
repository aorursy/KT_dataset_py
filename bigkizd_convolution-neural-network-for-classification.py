# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!pip install imutils
from imutils import paths
import cv2
def image_to_feature_vector(image, size=(32, 32)):
  return cv2.resize(image, size).flatten()

imagePaths = list(paths.list_images('../input/train/train/'))
print(imagePaths[0])
X_train = []
y_train = []
for (i, imagePath) in enumerate(imagePaths):
  image = cv2.imread(imagePath)
  label = imagePath.split('/')[-1].split('.')[0]
  pixels = image_to_feature_vector(image)
  X_train.append(pixels)
  y_train.append(label)
  if(i>0 and i%1000==0):
    print("[INFO] processed{}/{}".format(i, len(imagePaths)))
imagePaths = list(paths.list_images('../input/test1/test1/'))
print(imagePaths[0])
X_test = []
id_test = []
for (i, imagePath) in enumerate(imagePaths):
  image = cv2.imread(imagePath)
  idt = imagePath.split('/')[-1].split('.')[0]
  pixels = image_to_feature_vector(image)
  X_test.append(pixels)
  id_test.append(idt)
  if(i>0 and i%1000==0):
    print("[INFO] processed{}/{}".format(i, len(imagePaths)))
import numpy as np
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

labels = np.zeros((y_train.shape[0], 2))
for i in range(0, y_train.shape[0]):
  label = np.zeros(2)
  if(y_train[i]=='dog'):
    label[1]=1.0
  else:
    label[0]=1.0
  labels[i] = label

y_train=labels

# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
import tensorflow as tf
import pandas as pd
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10


num_input = 3072
num_classes = 2
dropout = 0.75

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)

def maxpool2d(x, k=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
def conv_net(x, weights, biases, dropout):
  x = tf.reshape(x, shape=[-1, 32, 32, 3])
  conv1 = conv2d(x, weights['wc1'], biases['bc1'])
  conv1 = maxpool2d(conv1, k=2)
  
  conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
  conv2 = maxpool2d(conv2, k=2)
  
  fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
  fc1 = tf.nn.dropout(fc1, dropout)
  
  out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
  return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}
biases = {
     'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for step in range(1, 250+1):
    batch_x = X_train[(step-1)*100:step*100]
    batch_y = y_train[(step-1)*100:step*100]
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
    if step % display_step == 0 or step == 1:
      loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
      print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
  print("Optimization Finished!")
  preds = sess.run(prediction, feed_dict={X:X_test, keep_prob:1.0})
  submit = pd.DataFrame()
  submit['id']=id_test
  label = []
  for i in range(0, preds.shape[0]):
    for j in range(0, preds.shape[1]):
      if(preds[i, j]==1):
        label.append(j)
  submit['label']= label
submit.to_csv('submit.csv', index=None)
