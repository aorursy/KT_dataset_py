import tensorflow as tf

import numpy as np



def conv2d(input, kernel_size, out_channels, strides, padding='SAME', relu=True, name=None):

    '''

    Create 2D convolutional layer

    :param input: A 4-D tensor with shape [batch, in_height, in_width, in_channels]

    :param kernel_size: A tuple of ints. Kernel/filter size

    :param out_channels: A int. Number of output channels

    :param strides: A tuple of ints. The stride of the sliding window

    :param padding: A string. The type of padding algorithm to use: 'SAME' or 'VALID'

    :param name: A string. A name for the layer

    :return: Tuple of layer and weights tensors

    '''

    # Get input shape

    input_shape = input.get_shape().as_list()



    with tf.variable_scope(name):

        # Define weights and bias

        weights = tf.Variable(tf.truncated_normal(shape=[*kernel_size, input_shape[3], out_channels], mean=0, stddev=0.01))

        bias = tf.Variable(tf.zeros(out_channels))

        # Build convolution layer

        conv_layer = tf.nn.conv2d(input, filter=weights, strides=[1, *strides, 1], padding=padding)

        # Add bias

        conv_layer = tf.nn.bias_add(conv_layer, bias)

        if relu:

            # Apply activation function

            conv_layer = tf.nn.relu(conv_layer)



    return (conv_layer, weights)