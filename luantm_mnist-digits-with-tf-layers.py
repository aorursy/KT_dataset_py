import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
Data = np.load("../input/mnist.npz")
x_train = Data['x_train'] / 255
x_test = Data['x_test'] / 255

y_train_cls = Data['y_train']
y_train = np.zeros((y_train_cls.shape[0] , 10))
y_train[np.arange(y_train_cls.shape[0]), y_train_cls] = 1

y_test_cls = Data['y_test']
y_test = np.zeros((y_test_cls.shape[0] , 10))
y_test[np.arange(y_test_cls.shape[0]), y_test_cls] = 1
print(x_train.shape)
print(y_train.shape)
print(y_train)
print(y_train_cls.shape)
# The number of pixels in each dimension of an image.
img_size = 28

# The images are stored in one-dimensional arrays of this length.
img_size_flat = 28 * 28

# Tuple with height and width of images used to reshape arrays.
img_shape = (28, 28)

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
x = tf.placeholder(tf.float32, shape=[None, 28, 28], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
input = x_image
layer_conv1 = tf.layers.conv2d(inputs=input, padding='same', filters=16,
                                  kernel_size=5, activation=tf.nn.relu)
layer1 = tf.layers.max_pooling2d(inputs=layer_conv1, pool_size=2, strides=2)

layer_conv2 = tf.layers.conv2d(inputs=layer1, padding='same', filters=36,
                                  kernel_size=5, activation=tf.nn.relu)
layer2 = tf.layers.max_pooling2d(inputs=layer_conv2, pool_size=2, strides=2)

flatten_layer = tf.contrib.layers.flatten(layer2)
dense1 = tf.layers.dense(inputs=flatten_layer, units=128, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense1, units=num_classes, activation=None)

logits = dense2
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        m  =x_train.shape[0]
        permutation = np.random.permutation(m)
        
        j = 0
        while j < m:
            k = j+64
            if k >=m:
                k = m - 1
            r = range(j, k)
            x_batch = x_train[permutation[r]]
            y_true_batch = y_train[permutation[r]]
        
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            session.run(optimizer, feed_dict={x: x_batch,
                                              y_true: y_true_batch})
            print('_________ : {}/{}: '.format(j, m))
            print_accuracy()
            
            j+=64
        print('iter : ', i)
        print_accuracy()
def print_accuracy():
    acc, y_pred_test_cls  = session.run([accuracy, y_pred_cls], feed_dict={x: x_test,
                                           y_true: y_test,
                                           y_true_cls: y_test_cls
                                          })
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
    return y_pred_test_cls
session = tf.Session()
s = session.run(tf.global_variables_initializer())
optimize(num_iterations=1)
y_pred_test_cls = print_accuracy()
# optimize(num_iterations=1)
# y_pred_test_cls = print_accuracy()
