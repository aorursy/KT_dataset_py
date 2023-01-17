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
def conv_2d(bottom, out_dim):

  _filter = tf.Variable(tf.random.normal(out_dim))

  conv = tf.nn.conv2d(bottom, _filter, strides=[1, 1, 1, 1], padding='SAME')

  return tf.nn.relu(conv)



def max_pool(bottom):

  return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def plot_images_labels_prediction(images, labels, prediction, idx, num=25):

  plt.figure(figsize=(10, 10))

  for i in range(num):

    plt.subplot(5, 5, i + 1)

    plt.imshow(images[idx].reshape(28, 28), cmap='binary')



    title = str(np.argmax(labels[idx]))

    if len(prediction) > 0:

      title += ', predict={}'.format(prediction[idx])



    plt.gca().set_title(title, fontsize=15)

    plt.gca().set_xticks([])

    plt.gca().set_yticks([])

    idx += 1

  plt.show()
def to_label(data):

  labels = list()



  for row in data:

    temp = [0.0] * 10

    temp[row] = 1.0

    

    labels.append(temp)

  

  return np.asarray(labels)
import tensorflow as tf

import tensorflow.keras.datasets.mnist as mnist

import numpy as np

import matplotlib.pyplot as plt

from time import time
tf.compat.v1.disable_eager_execution()



(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train, y_test = to_label(y_train), to_label(y_test)



print(x_train.shape)

x_train = x_train.reshape([-1, 28, 28, 1]) / 255.0

x_test = x_test.reshape([-1, 28, 28, 1]) / 255.0



train_data = DataSet(x_train, y_train)

# test_data = DataSet(x_test, y_test)



X = tf.compat.v1.placeholder('float', [None, 28, 28, 1])

y_label = tf.compat.v1.placeholder('float', [None, 10])
h1 = conv_2d(X, [3, 3, 1, 32])

h1 = max_pool(h1)

print('layer1 shape:', h1.get_shape())



h2 = conv_2d(h1, [3, 3, 32, 64])

h2 = max_pool(h2)

print('layer2 shape:', h2.get_shape())



h3 = conv_2d(h2, [3, 3, 64, 128])

h3 = max_pool(h3)

h3 = tf.reshape(h3, [-1, 128 * 4 * 4])

print('layer3 shape:', h3.get_shape())



h4 = tf.nn.relu(tf.matmul(h3, tf.Variable(tf.random.normal([128 * 4 * 4, 625]))))

print('layer4 shape:', h4.get_shape())



y_predict = tf.matmul(h4, tf.Variable(tf.random.normal([625, 10])))
loss_func = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=y_label))



op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_func)



correct_pre = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))



accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))
epochs = 200

batch_size = 100

totalBatchs = int(x_train.shape[0] / batch_size)



loss_list = list()

epoch_list = list()

acc_list = list()



print('start')

startTime = time()



sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(epochs):

    for i in range(totalBatchs):

        batch_x, batch_y = train_data.next_batch(batch_size)

        sess.run(op, feed_dict={X: batch_x, y_label: batch_y})



    loss, acc = sess.run([loss_func, accuracy], feed_dict={X: x_test, y_label: y_test})



    epoch_list.append(epoch)

    loss_list.append(loss)

    acc_list.append(acc)

    print('Train Epoch: {:02d}'.format(epoch + 1), 'Loss= {:.9f}'.format(loss), 'Accuracy= {}'.format(acc))

duration = time() - startTime

print('Train Finished takes:', duration)
fig = plt.gcf()

fig.set_size_inches(8, 4)

plt.plot(epoch_list, loss_list, label='Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Loss'], loc='upper left')

plt.show()
plt.plot(epoch_list, acc_list, label='Accuracy')

fig2 = plt.gcf()

fig2.set_size_inches(8, 4)

plt.ylim(0.92, 1)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, y_label: y_test}))



prediction_result = sess.run(tf.argmax(y_predict, 1), feed_dict={X: x_test})

print(prediction_result[:10])
plot_images_labels_prediction(x_test, y_test, prediction_result, 0)