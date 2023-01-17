import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf



# settings

LEARNING_RATE = 1e-4



# set to 20000 on local environment to get 0.99 accuracy

TRAINING_ITERATIONS = 20000



DROPOUT = 0.5

BATCH_SIZE = 100



# set to 0 to train on all available data

VALIDATION_SIZE = 2000



# image number to output

IMAGE_TO_DISPLAY = 10



# read training data from CSV file 

data = pd.read_csv('../input/train.csv')

images = data.iloc[:,1:].values



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



# in this case all images are square

image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



# display image

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# output image

display(images[IMAGE_TO_DISPLAY])



# labels

labels_flat = data[[0]].values.ravel()



labels_count = np.unique(labels_flat).shape[0]



def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print(labels.shape)

print(images.shape)



# split data into training & validation

validation_images = images[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = images[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]



print(validation_images.shape)

print(train_images.shape)
epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



# serve data by batches

def next_batch(batch_size):

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed



    start = index_in_epoch

    index_in_epoch += batch_size



    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1



        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]



        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch



    return train_images[start:end], train_labels[start:end]



def train():

    # 构建多层CNN模型

    

    # 输入placeholders(x, _y)

    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32, [None, 784], name='x-input')

        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')



    # 对输入x进行reshape, 变为28*28矩阵

    with tf.name_scope('input_reshape'):

        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])



    # 不能将weight和bias初始化为0

    def weight_variable(shape):

        """对权重进行合理的初始化"""

        initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial)



    def bias_variable(shape):

        """对偏差进行合理的初始化"""

        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)



    # 卷积与池化函数

    # padding SAME/VALID

    def conv2d(x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



    def max_pool_2x2(x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                              strides=[1, 2, 2, 1], padding='SAME')



    def nn_layer(input_tensor, con_width, con_heigh, input_dim, output_dim, layer_name, act=tf.nn.relu):

        """

        可重使用的简单神经网络层



        包括：w*x+b, Conv, ReLU, MaxPooling

        It also sets up name scoping so that the resultant graph is easy to read,

        and adds a number of summary ops.

        """

        # 为NNLayer增加一个name_scope以确保NN Graph中层与层之间的逻辑性

        with tf.name_scope(layer_name):

            # 设置并保存权重的状态

            with tf.name_scope('weights'):

                weights = weight_variable([con_width, con_heigh,  input_dim, output_dim])

        

            # 设置并保存偏差的状态

            with tf.name_scope('biases'):

                biases = bias_variable([output_dim])



            # 计算conv2d(w*x)+b、ReLU、Maxpooling，然后设置histogram_summary

            with tf.name_scope('Wx_plus_b'):

                preactivate = conv2d(input_tensor, weights) + biases

            # ReLU

            activations = act(preactivate, name='activation')

            # Maxpooling

            maxpooling = max_pool_2x2(activations)

        

            return maxpooling 

    

    # 构建两层卷积神经网络Conv+ReLU+Maxpooling

    # 1*28*28 --> 32*28*28 --> 32*14*14

    hidden_pool1 = nn_layer(image_shaped_input, 5, 5, 1, 32, 'layer_1')

    # 32*14*14 --> 64*14*14 --> 64*7*7

    hidden_pool2 = nn_layer(hidden_pool1, 5, 5, 32, 64, 'layer_2')

    

    # 密集链接层Matmul + ReLU

    # 64*7*7 --> 3136 --> 1024

    # 初始化权重与偏值

    with tf.name_scope('layer_fc1'):

        layer_name = 'layer_fc1'

        w_fc1 = weight_variable([7 * 7 * 64, 1024])

        b_fc1 = bias_variable([1024])

        

        #Reshape + Matmul + ReLU

        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])

        h_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, w_fc1) + b_fc1)

    

    # Dropout层

    #1024 --> 1024

    with tf.name_scope('dropout'):

        keep_prob = tf.placeholder(tf.float32)

        dropped = tf.nn.dropout(h_fc1, keep_prob)



    # Softmax层

    # 1024 --> 10

    with tf.name_scope('layer_fc2'):

        w_fc2 = weight_variable([1024, 10])

        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(dropped, w_fc2) + b_fc2)



    # 计算Cross_entropy

    with tf.name_scope('cross_entropy'):

        diff = y_ * tf.log(y_conv)

    with tf.name_scope('total'):

        cross_entropy = -tf.reduce_mean(diff)



    # AdamOptimizer训练

    with tf.name_scope('train'):

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    

    # 计算Accuracy

    with tf.name_scope('accuracy'):

        with tf.name_scope('correct_prediction'):

            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope('accuracy'):

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()

    sess.run(init)



    def feed_dict():

        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

        xs, ys = next_batch(BATCH_SIZE)

        k = DROPOUT

        

        return {x: xs, y_: ys, keep_prob: k}

    

    for i in range(TRAINING_ITERATIONS):

        if i % 100 == 0:

            acc = sess.run(accuracy, feed_dict=feed_dict())

            print('Accuracy at step %s: %s' % (i, acc))



if __name__ == '__main__':

    train()