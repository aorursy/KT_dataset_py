import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf
RANDOM_SEED = 98589741 # randomly chosen

CHECK_IMAGE_VALID = 185

CHECK_IMAGE_TRAIN = 8585

VALIDATION_PERCENT = 5
data = pd.read_csv('../input/train.csv')
images = data.iloc[:, 1:].values

images = images.astype(np.float)

images = np.multiply(images, 1 / 255.0)



labels_dense = data[[0]].values.ravel()



num_images = images.shape[0]

image_size = images.shape[1]



image_width = image_height = np.sqrt(image_size).astype(np.uint8)



validation_size = int(num_images * VALIDATION_PERCENT / 100.0 + 0.5)

train_size = num_images - validation_size



print ('There are {0} images of {1} = {2}x{3} pixels'.format(

    num_images, image_size, image_width, image_height))

print ("We'll use {0} for training and {1} for validation".format(train_size, validation_size))
def dense_to_one_hot(dense, classes):

    labels = dense.shape[0]

    offset = np.arange(labels) * classes

    one_hot = np.zeros((labels, classes))

    one_hot.flat[offset + dense.ravel()] = 1

    return one_hot



print (dense_to_one_hot)
labels_count = np.unique(labels_dense).shape[0]

labels = dense_to_one_hot(labels_dense, labels_count)

print ('Labels: {0}'.format(labels.shape))
np.random.seed(RANDOM_SEED)



train_mask = np.ones(num_images, dtype=bool)

valid_indices = np.random.choice(num_images, validation_size, False)

train_mask[valid_indices] = False

validation_mask = np.invert(train_mask)



print(train_mask[0:10])

print(validation_mask[:10])

train_images = images[train_mask]

valid_images = images[validation_mask]

train_labels = labels[train_mask]

valid_labels = labels[validation_mask]



print('train: {0} ... {1}'.format(train_images.shape, train_labels.shape))

print('valid: {0} ... {1}'.format(valid_images.shape, valid_labels.shape))
def display(index):

    img = images[index]

    image = img.reshape(image_width, image_height)

    plt.axis('off')

    plt.imshow(image, cmap = cm.binary)

    print ('Label: {0} ({1})'.format(labels_dense[index], train_mask[index]))

    

print (display)
display(CHECK_IMAGE_TRAIN)
display(CHECK_IMAGE_VALID)
def create_variable(shape, variable_fn):

    initial = variable_fn(shape)

    return tf.Variable(initial)
def truncated_normal(shape):

    return tf.truncated_normal(shape, stddev = 0.1)



def constant(shape):

    return tf.constant(0.1, shape = shape)
def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides = [1] * 4, padding = 'SAME')
def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
WEIGHT_VARIABLE_FN = truncated_normal

BIAS_VARIABLE_FN = constant

CONV_FN = conv2d

POOL_FN = max_pool_2x2
def weight_variable(shape):

    return create_variable(shape, WEIGHT_VARIABLE_FN)



def bias_variable(shape):

    return create_variable(shape, BIAS_VARIABLE_FN)



def convolution(x, W):

    return CONV_FN(x, W)



def pool(x):

    return POOL_FN(x)
# images

x = tf.placeholder('float', shape = [None, image_size])



# labels

y_ = tf.placeholder('float', shape = [None, labels_count])



# dropout

keep_prob = tf.placeholder('float')
def create_convolution_layer(x, features, patch_width = 5, patch_height = 5):

    W = weight_variable([patch_width, patch_height, x.get_shape().as_list()[-1], features])

    b = bias_variable([features])



    h = tf.nn.relu(convolution(x, W) + b)

    y = pool(h)



    return y



def create_dense_layer(x, units, keep_prob):

    W = weight_variable([x.get_shape().as_list()[-1], units])

    b = bias_variable([units])

    

    h = tf.nn.relu(tf.matmul(x, W) + b)

    d = tf.nn.dropout(h, keep_prob)

    

    return d



def create_softmax_layer(x, labels):

    W = weight_variable([x.get_shape().as_list()[-1], labels])

    b = bias_variable([labels])

    

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    return y



def create_network(input, convolutional_layers, hidden_layers, labels_count, keep_prob):

    network = [input]



    # create the convolutional layers

    for features in convolutional_layers:

        network.append(create_convolution_layer(network[-1], features))



    # flatten the last convolutional layer

    if len(convolutional_layers) > 0:

        flat_size = np.prod(network[-1].get_shape().as_list()[1:])

        network.append(tf.reshape(network[-1], [-1, flat_size]))

        

    # add the dense layers

    for units in hidden_layers:

        network.append(create_dense_layer(network[-1], units, keep_prob))

        

    # add a readout softmax layer

    network.append(create_softmax_layer(network[-1], labels_count))

    

    return network
network = create_network(

    input = tf.reshape(x, [-1, image_width, image_height, 1], name = 'InputImage'),

    convolutional_layers = [32, 64],

    hidden_layers = [1024],

    labels_count = labels_count,

    keep_prob = keep_prob)



print('Network:')

for layer in network:

    print('  {0}'.format(layer))
# final layer of network has the output

y = network[-1]



# cost function

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
LEARNING_RATE = 1e-3



# optimisation function

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)



# evaluation

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y, 1)
epochs_completed = 0

index_in_epoch = 0

permutation = np.random.permutation(train_size)



def reset_epoch():

    global index_in_epoch

    global permutation



    permutation = np.random.permutation(train_size)

    index_in_epoch = 0



def next_batch(batch_size):

    global permutation

    global index_in_epoch

    global epochs_completed



    if index_in_epoch + batch_size > train_size:

        epochs_completed += 1

        reset_epoch()



    start = index_in_epoch

    index_in_epoch += batch_size



    if index_in_epoch > train_size:

        start = 0

        index_in_epoch = batch_size



    end = index_in_epoch

    return train_images[permutation[start:end]], train_labels[permutation[start:end]]        
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()



sess.run(init)
NUM_EPOCHS = 1

DROPOUT = 0.5

BATCH_SIZE = 50
train_accuracies = []

validation_accuracies = []

x_range = []



iteration = 0

display_step = 1



reset_epoch()



while epochs_completed < NUM_EPOCHS:

    batch_x, batch_y = next_batch(BATCH_SIZE)

    

    if iteration % display_step == 0:

        train_accuracy = accuracy.eval(feed_dict = {

            x: batch_x,

            y_: batch_y,

            keep_prob: 1.0})

        if VALIDATION_PERCENT > 0:

            validation_accuracy = accuracy.eval(feed_dict = {

                x: valid_images[0 : BATCH_SIZE],

                y_: valid_labels[0 : BATCH_SIZE],

                keep_prob: 1.0})

            print ('accuracy (train/validation) => %.2f/%.2f for step (global/epoch) %d/%d' % (train_accuracy, validation_accuracy, iteration, epochs_completed))

            validation_accuracies.append(validation_accuracy)

        else:

            print ('train accuracy => %.2f for step (global/epoch) %d/%d' % (train_accuracy, iteration, epochs_completed))

        train_accuracies.append(train_accuracy)

        x_range.append(iteration)

        

        if iteration % (display_step * 10) == 0 and iteration > 0:

            display_step *= 10



    # train on batch

    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: DROPOUT})

    iteration += 1

    

print ('Training finished after %d/%d' % (iteration, epochs_completed))
# Compute train accuracy on all set

print ('Computing overall accuracy')



reset_epoch()

s = epochs_compeleted

overall_accuracy = 0

iteration = 0

display_step = 1

while epochs_compeleted == s:

    batch_x, batch_y = next_batch(BATCH_SIZE)

    train_accuracy = accuracy.eval(feed_dict = {

            x: batch_x,

            y_: batch_y,

            keep_prob: 1.0})

    print ('Batch %d accuracy %.2f' % (iteration, train_accuracy))

    overall_accuracy += train_accuracy

    iteration += 1

    

    if iteration % (display_step * 10) == 0 and iteration > 0:

        display_step *= 10



overall_accuracy /= iteration

print ('Final train accuracy: %.4f' % train_accuracy)
# check final accuracy on validation set  

if VALIDATION_PERCENT > 0:

    validation_accuracy = accuracy.eval(feed_dict={x: valid_images, 

                                                   y_: valid_labels, 

                                                   keep_prob: 1.0})

    print('validation_accuracy => %.4f' % validation_accuracy)

    plt.plot(x_range, train_accuracies,'-b', label='Training')

    plt.plot(x_range, validation_accuracies,'-g', label='Validation')

    plt.legend(loc='lower right', frameon=False)

    plt.ylim(ymax = 1.1, ymin = 0.7)

    plt.ylabel('accuracy')

    plt.xlabel('step')

    plt.show()