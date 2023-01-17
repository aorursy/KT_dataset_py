import tensorflow as tf



def fully_conn(input, num_outputs, name=None):

    '''

    Create a fully-connected layer

    :param input A 2-D tensor

    :param num_outputs Number of outputs

    :param name: A string. A name for the layer

    :return: Tuple of layer and weights tensors

    '''

    # Get input shape

    input_shape = input.get_shape().as_list()



    with tf.variable_scope(name):

        # Define weights and bias

        weights = tf.Variable(tf.truncated_normal(shape=[input_shape[1], num_outputs], mean=0, stddev=0.01))

        bias = tf.Variable(tf.zeros(num_outputs))

        # Apply matrix multiplication

        fully_conn_layer = tf.add(tf.matmul(input, weights), bias)

        # Apply activation function

        fully_conn_layer = tf.nn.relu(fully_conn_layer)



    return (fully_conn_layer, weights)