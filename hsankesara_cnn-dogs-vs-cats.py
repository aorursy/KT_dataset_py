import tensorflow as tf
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
prefix = '../input/dog vs cat/dataset/training_set/'
os.listdir('../input/dog vs cat/dataset/training_set')
classes = ['dog', 'cat']
num_classes = 1
limit = 1000
df_train_dog = np.array([None] * limit)
j = 0
for i in os.listdir(prefix + 'dogs/'):
    if j < limit:
        df_train_dog[j] = plt.imread(prefix + 'dogs/' + i)
        j += 1
    else:
        break
df_train_dog_op = np.zeros(limit)
df_train_cat = np.array([None] * limit)
j = 0
for i in os.listdir(prefix + 'cats/'):
    if j < limit:
        df_train_cat[j] = plt.imread(prefix + 'cats/' + i)
        j += 1
    else:
        break
df_train_cat_op = np.ones(limit)
size = (128, 128)
num_channels = 3
df_train = np.concatenate((df_train_cat, df_train_dog))
df_train_o = np.concatenate((df_train_cat_op,df_train_dog_op))
del(df_train_cat)
del(df_train_dog)
del(df_train_cat_op)
del(df_train_dog_op)
j = 0
for i in df_train:
    df_train[j] = cv2.resize(i, size)
    df_train[j] = df_train[j].reshape(1, size[0], size[1], num_channels)
    j += 1

from sklearn.utils import shuffle
df_train, df_train_o = shuffle(df_train, df_train_o)
df_train = np.vstack(df_train[:])
df_train_o = df_train_o.reshape(2 * limit, 1)
df_train.shape
from sklearn.model_selection import train_test_split
df_train, df_test, df_train_o, df_test_o = train_test_split(df_train, df_train_o, test_size = 0.2)
beta = 0.01
def create_weights(shape, suffix):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.03), name='W_' + suffix)

def create_biases(size, suffix):
    return tf.Variable(tf.truncated_normal([size]), name='b_' + suffix)
def conv_layer(inp, kernel_shape, num_channels, num_kernels, suffix):
    filter_shape = [kernel_shape[0], kernel_shape[1], num_channels, num_kernels]
    weights = create_weights(shape=filter_shape, suffix=suffix)
    biases = create_biases(num_kernels, suffix=suffix)
    layer = tf.nn.conv2d(input=inp, filter=weights, padding='SAME', strides=[1, 1, 1, 1], name='conv_' + suffix)
    layer += biases
    layer = tf.nn.relu6(layer, name='relu_' + suffix)
    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
    return layer
def flatten_layer(layer, suffix):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features], name='flat_' + suffix )
    return layer
def dense_layer(inp, num_inputs, num_outputs, suffix, use_relu=True):
    weights = create_weights([num_inputs, num_outputs], suffix)
    biases = create_biases(num_outputs, suffix)
    layer = tf.matmul(inp, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    layer = tf.nn.batch_normalization(layer, mean=0, variance=1, scale=True, offset=0, variance_epsilon=0.01)
        
    return layer
tf.device("/device:GPU:0")
x = tf.placeholder(tf.float32, shape=[None, size[0],size[1],num_channels], name='x')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
layer_conv1 = conv_layer(inp=x, kernel_shape=(3, 3), num_kernels=32, num_channels=3, suffix='1')
layer_conv2 = conv_layer(inp=layer_conv1, kernel_shape=(3, 3), num_kernels=64, num_channels=32, suffix='2')
layer_conv3 = conv_layer(inp=layer_conv2, kernel_shape=(64, 64), num_kernels=32, num_channels=64, suffix='3')
flat_layer = flatten_layer(layer_conv3, suffix='3')
dense_layer_1 = dense_layer(inp=flat_layer, num_inputs=8192, num_outputs=4096, suffix='4')
dense_layer_2 = dense_layer(inp=dense_layer_1, num_inputs=4096, num_outputs=num_classes, suffix='5', use_relu=False)
y_ = tf.nn.sigmoid(dense_layer_2)
w4 = tf.nn.l2_loss([v for v in tf.global_variables() if v.name == "W_4:0"][0])
w5 = tf.nn.l2_loss([v for v in tf.global_variables() if v.name == "W_5:0"][0])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=dense_layer_2))
regularizer = w5  + w4
cost = tf.reduce_mean(cost + beta * regularizer )
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
# define an accuracy assessment operation
prediction = tf.round(y_)
correct_prediction = tf.equal(y, prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
epochs = 60
batch_size =  64
display = 10
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    total_batch = df_train.shape[0] // batch_size
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = df_train[i * batch_size : (i + 1) * batch_size]
            batch_y = df_train_o[i * batch_size : (i + 1) * batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={x: df_train, y: df_train_o})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " train accuracy: {:.3f}".format(test_acc))
        if epoch % display == 0:
            print('Test Accuracy after epoch ', (epoch + 1), ': ', sess.run(accuracy, feed_dict={x: df_test, y: df_test_o}))
    print("\nTraining complete!")
    print('Test Accuracy: ', sess.run(accuracy, feed_dict={x: df_test, y: df_test_o}))
    save_path = saver.save(sess, "../working/model.ckpt")



