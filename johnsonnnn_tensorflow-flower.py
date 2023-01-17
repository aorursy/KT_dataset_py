class DataSet(object):

  def __init__(self, images, labels):

    self.images = images

    self.labels = labels



    self._images = images

    self._labels = labels

    self._epochs_completed = 0

    self._index_in_epoch = 0

    self._num_examples = len(images)

  

  def to_shuffle(self):

    perm = np.arange(self._num_examples)

    np.random.shuffle(perm)

    self._images = self.images[perm]

    self._labels = self.labels[perm]



  def next_batch(self, batch_size, fake_data=False, shuffle=True):

    start = self._index_in_epoch



    if start == 0 and self._epochs_completed == 0 and shuffle:

      self.to_shuffle()

    

    if start + batch_size > self._num_examples:   #this end with next start

      self._epochs_completed += 1



      rest_num_examples = self._num_examples - start

      images_rest_part = self._images[start : self._num_examples]

      labels_rest_part = self._labels[start : self._num_examples]



      if shuffle:

        self.to_shuffle()

      

      start = 0

      self._index_in_epoch = batch_size - rest_num_examples

      end = self._index_in_epoch

      images_new_part = self._images[start : end]

      labels_new_part = self._labels[start : end]



      return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)



    else:

      self._index_in_epoch += batch_size

      end = self._index_in_epoch

      return self._images[start : end], self._labels[start : end]
def to_label(data):

  labels = list()



  for row in data:

    temp = [0.0] * 3

    temp[row] = 1.0

    

    labels.append(temp)

  

  return np.asarray(labels)



def layer(input_dim, output_dim, inputs, activation=None):

  W = tf.Variable(tf.random.normal([input_dim, output_dim]))

  b = tf.Variable(tf.random.normal([1, output_dim]))

  XWb = tf.matmul(inputs, W) + b



  if activation:

    outputs = activation(XWb)

  else:

    outputs = XWb

  return outputs
import os

mainPath = '/kaggle/input'

trainPath = os.path.join(mainPath, 'iris_training.csv')

testPath = os.path.join(mainPath, 'iris_test.csv')
import pandas as pd

import tensorflow as tf

import numpy as np



CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth', 'Species']

train_data = pd.read_csv(trainPath, names=CSV_COLUMN_NAMES, header=0)

test_data = pd.read_csv(testPath, names=CSV_COLUMN_NAMES, header=0)

print(train_data)

train_data.head()
flower0 = train_data[train_data.loc[:, 'Species'] == 0]

flower1 = train_data[train_data.loc[:, 'Species'] == 1]

flower2 = train_data[train_data.loc[:, 'Species'] == 2]



# print(flower1)

flower0.plot(kind='scatter', x='PetalLength', y='PetalWidth')

flower1.plot(kind='scatter', x='PetalLength', y='PetalWidth')

flower2.plot(kind='scatter', x='PetalLength', y='PetalWidth')
ax = flower0.plot(kind='scatter', x='PetalLength', y='PetalWidth')

flower1.plot(kind='scatter', x='PetalLength', y='PetalWidth', ax=ax)

flower2.plot(kind='scatter', x='PetalLength', y='PetalWidth', ax=ax)
ax = flower0.plot(kind='scatter', x='PetalLength', y='PetalWidth', color='red', label='Setosa', figsize=(10, 6))

flower1.plot(kind='scatter', x='PetalLength', y='PetalWidth', color='green', label='Versicolor', ax=ax)

flower2.plot(kind='scatter', x='PetalLength', y='PetalWidth', color='blue', label='Virginica', ax=ax)
train_data.SepalLength.plot(kind='box')

train_data[['PetalWidth', 'Species']].boxplot(by='Species', figsize=(10, 6))

train_data.describe()

x_train, y_train = train_data, train_data.pop('Species')

x_test, y_test = test_data, test_data.pop('Species')

x_train.head()
y_train = to_label(y_train)

y_test = to_label(y_test)

print(y_train[0])
tf.compat.v1.disable_eager_execution()



X = tf.compat.v1.placeholder('float', [None, 4])

y_label = tf.compat.v1.placeholder('float', [None, 3])
h1 = layer(4, 10, X, tf.nn.relu)

h2 = layer(10, 10, h1, tf.nn.relu)

h3 = layer(10, 10, h2, tf.nn.relu)

y_predict = layer(10, 3, h3)
loss_func = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=y_label))



op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_func)



correct_pre = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))



accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))
from sklearn import preprocessing



x_train = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(x_train.values)

x_test = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(x_test.values)

print(x_train[0])

print()

print(x_test[0])
_train_data = DataSet(x_train, y_train)
epochs = 200

batch_size = 10

totalBatchs = int(x_train.shape[0] / batch_size)



loss_list = list()

epoch_list = list()

acc_list = list()



from time import time



startTime = time()





sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())



for epoch in range(epochs):

    for i in range(totalBatchs):

        batch_x, batch_y = _train_data.next_batch(batch_size)

        sess.run(op, feed_dict={X: batch_x, y_label: batch_y})



    loss, acc = sess.run([loss_func, accuracy], feed_dict={X: x_test, y_label: y_test})



    epoch_list.append(epoch)

    loss_list.append(loss)

    acc_list.append(acc)

    print('Train Epoch: {:02d}'.format(epoch + 1), 'Loss= {:.9f}'.format(loss), 'Accuracy= {}'.format(acc))

duration = time() - startTime

print('Train Finished takes:', duration)
import matplotlib.pyplot as plt



fig = plt.gcf()

fig.set_size_inches(10, 5)



plt.plot(epoch_list, loss_list, label='Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
fig2 = plt.gcf()

fig2.set_size_inches(10, 5)



# plt.xlim(0, 80)

# plt.ylim(0.8, 1)

plt.plot(epoch_list, acc_list, label='Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, y_label: y_test}))
prediction_result = sess.run(tf.argmax(y_predict, 1), feed_dict={X: x_test})

print('Prediction:', prediction_result[:10])

print('Answer    :', sess.run(tf.argmax(y_test[:10], 1)))