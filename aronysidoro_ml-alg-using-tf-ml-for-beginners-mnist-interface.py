import tensorflow as tf 

import numpy as np

import pandas as pd
def label_vector_to_one_hot_vector(vector, one_hot_size=10):

    """

    Use to convert a column vector to a 'one-hot' matrix

    

    Example:

        vector: [[2], [0], [1]]

        one_hot_size: 3

        returns:

            [[ 0.,  0.,  1.],

             [ 1.,  0.,  0.],

             [ 0.,  1.,  0.]]

    

    Parameters:

        vector (np.array): of size (n, 1) to be converted

        one_hot_size (int) optional: size of 'one-hot' row vector

        

    Returns:

        np.array size (vector.size, one_hot_size): converted to a 'one-hot' matrix

    """

    squeezed_vector = np.squeeze(vector, axis=-1)



    one_hot = np.zeros((squeezed_vector.size, one_hot_size))



    one_hot[np.arange(squeezed_vector.size), squeezed_vector] = 1

    

    return one_hot
class DataSet(object):

    

    def __init__(self, df, train_dev_slice):

        labels = np.array(df['label'].values[train_dev_slice])

        self._labels = labels.reshape(labels.shape[0], 1)

        

        self._images = df.drop(['label'], axis=1).values[train_dev_slice]

        

        self._current_batch = 0

        

    def next_batch(self, n):

        start = n * self._current_batch

        end = n * (self._current_batch + 1)

        # increment for next mini-batch

        self._current_batch += 1

        return self.images[start:end], self.labels[start:end]

    

    @property

    def labels(self):

        return label_vector_to_one_hot_vector(self._labels)



    @property

    def images(self):

        return self._images





class Mnist(object):

    

    def __init__(self, df, train_dev_slice):

        # df = pd.read_csv(csv_file_path)

        

        self._train = DataSet(df, train_dev_slice=slice(0, train_dev_slice))

        size = df['label'].size # 42000

        self._test = DataSet(df, train_dev_slice=slice(train_dev_slice, size))

                    

    @property

    def train(self):

        return self._train



    @property

    def test(self):

        return self._test
# Import data

df = pd.read_csv('../input/train.csv')



mnist = Mnist(df, 40000)



# Create the model

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b



# Define loss and optimizer

y_ = tf.placeholder(tf.float32, [None, 10])



cross_entropy = tf.reduce_mean(

      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# Train

for idx in range(1000):

    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if idx % 100 == 0:

        # Test trained model

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))



        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('Iteration: %s' % idx, sess.run(accuracy, feed_dict={x: mnist.test.images,

                                          y_: mnist.test.labels}))
test_df = pd.read_csv('../input/test.csv')

test_data = test_df.values



test_pred = sess.run(y, feed_dict={x: test_data})

test_labels = np.argmax(test_pred, axis=1)
submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})

submission.to_csv('submission.csv', index=False)

submission.tail()