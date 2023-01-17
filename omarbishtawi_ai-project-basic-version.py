import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
cifar10_dataset_folder_path = '../input/cifar10/cifar-10-batches-py'
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import sys

sys.path.append('../input/aiproject/')



import helper

import numpy as np



# Explore the dataset

batch_id = 1

sample_id = 5

helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
def normalize(x):

    """

    Normalize a list of sample image data in the range of 0 to 1

    : x: List of image data.  The image shape is (32, 32, 3)

    : return: Numpy array of normalize data

    """

    # TODO: Implement Function

    #Normalization equation

    #zi=xi−min(x)/max(x)−min(x)

    normalized = (x-np.min(x))/(np.max(x)-np.min(x))

    return normalized
def one_hot_encode(x):

    """

    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.

    : x: List of sample Labels

    : return: Numpy array of one-hot encoded labels

    """

    # TODO: Implement Function

    return np.eye(10)[x]
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
!pip install tensorflow==1.5.0
import tensorflow as tf

def neural_net_image_input(image_shape):

    """

    Return a Tensor for a bach of image input

    : image_shape: Shape of the images

    : return: Tensor for image input.

    """

    # TODO: Implement Function

    return tf.placeholder(tf.float32, shape=(None, *image_shape), name='x')



def neural_net_label_input(n_classes):

    """

    Return a Tensor for a batch of label input

    : n_classes: Number of classes

    : return: Tensor for label input.

    """

    # TODO: Implement Function

    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')



def neural_net_keep_prob_input():

    """

    Return a Tensor for keep probability

    : return: Tensor for keep probability.

    """

    # TODO: Implement Function

    return tf.placeholder(tf.float32, name='keep_prob')
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):

    """

    Apply convolution then max pooling to x_tensor

    :param x_tensor: TensorFlow Tensor

    :param conv_num_outputs: Number of outputs for the convolutional layer

    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer

    :param conv_strides: Stride 2-D Tuple for convolution

    :param pool_ksize: kernal size 2-D Tuple for pool

    :param pool_strides: Stride 2-D Tuple for pool

    : return: A tensor that represents convolution and max pooling of x_tensor

    """

    

    

    #Define weight

    weight_shape = [*conv_ksize, int(x_tensor.shape[3]), conv_num_outputs]

    w = tf.Variable(tf.random_normal(weight_shape, stddev=0.1))

    

    #Define bias

    b = tf.Variable(tf.zeros(conv_num_outputs))

    

    #Apply convolution

    x = tf.nn.conv2d(x_tensor, w, strides=[1, *conv_strides, 1], padding='SAME')

    

    #Apply bias

    x = tf.nn.bias_add(x, b)

    

    #Apply RELU

    x = tf.nn.relu(x)

    

    #Apply Max pool

    x = tf.nn.max_pool(x, [1, *pool_ksize, 1], [1, *pool_strides, 1], padding='SAME')

    return x
def flatten(x_tensor):

    """

    Flatten x_tensor to (Batch Size, Flattened Image Size)

    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.

    : return: A tensor of size (Batch Size, Flattened Image Size).

    """

    # TODO: Implement Function

    batch_size, *fltn_img_size = x_tensor.get_shape().as_list()

    img_size = fltn_img_size[0] * fltn_img_size[1] * fltn_img_size[2]

    tensor = tf.reshape(x_tensor, [-1, img_size])

    return tensor
def fully_conn(x_tensor, num_outputs):

    """

    Apply a fully connected layer to x_tensor using weight and bias

    : x_tensor: A 2-D tensor where the first dimension is batch size.

    : num_outputs: The number of output that the new tensor should be.

    : return: A 2-D tensor where the second dimension is num_outputs.

    """

    # TODO: Implement Function

    #weights

    w_shape = (int(x_tensor.get_shape().as_list()[1]), num_outputs)

    weights = tf.Variable(tf.random_normal(w_shape, stddev=0.1))

    

    #bias

    bias = tf.Variable(tf.zeros(num_outputs))

    x = tf.add(tf.matmul(x_tensor, weights), bias)

    output = tf.nn.relu(x)

    return output
def output(x_tensor, num_outputs):

    """

    Apply a output layer to x_tensor using weight and bias

    : x_tensor: A 2-D tensor where the first dimension is batch size.

    : num_outputs: The number of output that the new tensor should be.

    : return: A 2-D tensor where the second dimension is num_outputs.

    """

    # TODO: Implement Function

    #weights

    w_shape = (int(x_tensor.get_shape().as_list()[1]), num_outputs)

    weights = tf.Variable(tf.random_normal(w_shape, stddev=0.1))

    

    #bias

    bias = tf.Variable(tf.zeros(num_outputs))

    x = tf.add(tf.matmul(x_tensor, weights), bias)

    return x
def conv_net(x, keep_prob):

    """

    Create a convolutional neural network model

    : x: Placeholder tensor that holds image data.

    : keep_prob: Placeholder tensor that hold dropout keep probability.

    : return: Tensor that represents logits

    """

    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers

    x = conv2d_maxpool(x, 32, (3, 3), (1, 1), (2, 2), (2, 2))

    x = conv2d_maxpool(x, 32, (3, 3), (2, 2), (2, 2), (2, 2))

    x = conv2d_maxpool(x, 64, (3, 3), (1, 1), (2, 2), (2, 2))

        # TODO: Apply a Flatten Layer

    # Function Definition from Above:

    x = flatten(x)

        # TODO: Apply 1, 2, or 3 Fully Connected Layers

    #    Play around with different number of outputs

    # Function Definition from Above:

    #   fully_conn(x_tensor, num_outputs)

    x = fully_conn(x, 128)

    x = tf.nn.dropout(x, keep_prob)

    

    

    # TODO: Apply an Output Layer

    #    Set this to the number of classes

    # Function Definition from Above:

    #   output(x_tensor, num_outputs)

    result = output(x, 10)

    

    

    # TODO: return output

    return result

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""##############################

## Build the Neural Network ##

############################### Remove previous weights, bias, inputs, etc..

from tensorflow.python.framework import ops

ops.reset_default_graph()

print(tf.__version__)



# tf.reset_default_graph()# Inputs

x = neural_net_image_input((32, 32, 3))

y = neural_net_label_input(10)

keep_prob = neural_net_keep_prob_input()# Model

logits = conv_net(x, keep_prob)# Name logits Tensor, so that is can be loaded from disk after training

logits = tf.identity(logits, name='logits')# Loss and Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer().minimize(cost)# Accuracy

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):

    """

    Optimize the session on a batch of images and labels

    : session: Current TensorFlow session

    : optimizer: TensorFlow optimizer function

    : keep_probability: keep probability

    : feature_batch: Batch of Numpy image data

    : label_batch: Batch of Numpy label data

    """

    # TODO: Implement Function

    session.run(optimizer, feed_dict={

        x: feature_batch,

        y: label_batch,

        keep_prob: keep_probability

    })
def print_stats(session, feature_batch, label_batch, cost, accuracy):

    """

    Print information about loss and validation accuracy

    : session: Current TensorFlow session

    : feature_batch: Batch of Numpy image data

    : label_batch: Batch of Numpy label data

    : cost: TensorFlow cost function

    : accuracy: TensorFlow accuracy function

    """

     # TODO: Implement Function

    loss = session.run(cost, feed_dict={

        x: feature_batch,

        y: label_batch,

        keep_prob: 1.

    })

    

    valid_accuracy = session.run(accuracy, feed_dict={

        x: valid_features,

        y: valid_labels,

        keep_prob: 1.

    })



    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_accuracy))
# TODO: Tune Parameters

epochs = 52

batch_size = 256

keep_probability = 0.6
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

print('Checking the Training on a Single Batch...')

with tf.Session() as sess:

    # Initializing the variables

    sess.run(tf.global_variables_initializer())

    

    # Training cycle

    for epoch in range(epochs):

        batch_i = 1

        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):

            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

        print_stats(sess, batch_features, batch_labels, cost, accuracy)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import tensorflow as tf

import pickle

import helper

import random



# Set batch size if not already set

try:

    if batch_size:

        pass

except NameError:

    batch_size = 64



save_model_path = './image_classification'

n_samples = 4

top_n_predictions = 3



def test_model():

    """

    Test the saved model against the test dataset

    """



    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))

    loaded_graph = tf.Graph()



    with tf.Session(graph=loaded_graph) as sess:

        # Load model

        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        loader.restore(sess, save_model_path)



        # Get Tensors from loaded model

        loaded_x = loaded_graph.get_tensor_by_name('x:0')

        loaded_y = loaded_graph.get_tensor_by_name('y:0')

        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        

        # Get accuracy in batches for memory limitations

        test_batch_acc_total = 0

        test_batch_count = 0

        

        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):

            test_batch_acc_total += sess.run(

                loaded_acc,

                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})

            test_batch_count += 1



        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))



        # Print Random Samples

        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))

        random_test_predictions = sess.run(

            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),

            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})

        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)

test_model()