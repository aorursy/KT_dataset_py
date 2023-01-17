import tensorflow as tf

import pandas as pd

import numpy as np
data = pd.read_csv('../input/train.csv')



print('data({0[0]},{0[1]})'.format(data.shape))

print (data.head())
images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



print('images({0[0]},{0[1]})'.format(images.shape))
# read test data from CSV file 

test_images = pd.read_csv('../input/test.csv').values

test_images = test_images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

test_images = np.multiply(test_images, 1.0 / 255.0)



print('test_images({0[0]},{0[1]})'.format(test_images.shape))
labels_flat = data.iloc[:, 0].values.ravel()



print('labels_flat({0})'.format(len(labels_flat)))
labels_count = np.unique(labels_flat).shape[0] 



#shape[0] returns number of rows (from shape of an array) and shape[1] returns number of columns



print('labels_count => {0}'.format(labels_count))
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('labels({0[0]},{0[1]})'.format(labels.shape))
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#file_writer = tf.summary.FileWriter('../output', sess.graph)
train_images = images[2000:]

train_labels = labels[2000:]
for _ in range(1000):

  batch_xs, batch_ys = train_images, train_labels

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
prediction = tf.argmax(y,1)

predicted_labels = prediction.eval(feed_dict={x: test_images})

print(prediction.eval(feed_dict={x: test_images}))
np.savetxt('submission_softmax.csv', 

           np.c_[range(1,len(test_images)+1),predicted_labels], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')