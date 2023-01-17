import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import tensorflow as tf

import time

from datetime import timedelta
#loading the data sets from the csv files

print('--------load train & test file------')

# read training data from CSV file 

train_dataset = pd.read_csv('../input/train.csv')

print(train_dataset.shape)

train_dataset = train_dataset.as_matrix()



#test dataset

test_dataset = pd.read_csv('../input/test.csv')

print(test_dataset.shape)

test_dataset = test_dataset.as_matrix()

print("------finish loading --------------------")
validation_size = 2000

pixel_depth = 255.0  # Number of levels per pixel.

image_width = 28

image_height = 28
#Shuffle the datasets

from sklearn.utils import shuffle

train_dataset = shuffle(train_dataset)

print ('finish shuffling the data')
dataset_labels = train_dataset[: ,:1]

dataset_images = train_dataset[: ,1:]

print(dataset_labels.shape)

print(dataset_images.shape)
from sklearn.preprocessing import OneHotEncoder

# Fit the OneHotEncoder

enc = OneHotEncoder().fit(dataset_labels.reshape(-1, 1))

# Transform the label values to a one-hot-encoding scheme

dataset_labels = enc.transform(dataset_labels.reshape(-1, 1)).toarray()



print("Training set", dataset_labels.shape)

print ('labels[{0}] => {1}'.format(1,dataset_labels[1]))
del train_dataset
# split data into dataset_images & validation

X_val = dataset_images[:validation_size]

y_val = dataset_labels[:validation_size]



X_train = dataset_images[validation_size:]

y_train = dataset_labels[validation_size:]
### Let's apply normalization to our images.

X_train = (X_train.astype(float) - 

                    pixel_depth / 2) / pixel_depth



X_val = (X_val.astype(float) - 

                    pixel_depth / 2) / pixel_depth



X_test = (test_dataset.astype(float) - 

                    pixel_depth / 2) / pixel_depth
del dataset_images

del dataset_labels

del test_dataset
#reshpe the array 

X_train = X_train.reshape(-1, image_width, image_height, 1)

X_val =  X_val.reshape(-1, image_width, image_height, 1)

X_test =  X_test.reshape(-1, image_width, image_height, 1)



print('Training set', X_train.shape, y_train.shape)

print('Validation set', X_val.shape, y_val.shape)

print('Test set', X_test.shape)
def plot_images(images, nrows, ncols, cls_true, cls_pred=None):

    """ Plot nrows * ncols images from images and annotate the images

    """

    # Initialize the subplotgrid

    fig, axes = plt.subplots(nrows, ncols)

    

    # Randomly select nrows * ncols images

    rs = np.random.choice(images.shape[0], nrows*ncols)

    

    # For every axes object in the grid

    for i, ax in zip(rs, axes.flat): 

        

        # Predictions are not passed

        if cls_pred is None:

            title = "T: {0}".format(np.argmax(cls_true[i]))

        

        # When predictions are passed, display labels + predictions

        else:

            title = "T: {0}, P: {1}".format(np.argmax(cls_true[i]), cls_pred[i])  

            

        # Display the image

        ax.imshow(images[i,:,:,0], cmap='binary')

        #ax.imshow(images[i].reshape(image_width,image_height), cmap='binary')

        

        # Annotate the image

        ax.set_title(title)

        

        # Do not overlay a grid

        ax.set_xticks([])

        ax.set_yticks([])

    fig.show()
### Plot 2 rows with 9 images each from the training set

plot_images(X_train, 2, 9, y_train);
# We know that SVHN images have 32 pixels in each dimension

img_size = X_train.shape[1] #28

# Greyscale images only have 1 color channel

num_channels = X_train.shape[-1] # 1 gray

# Number of classes, one class for each of 10 digits

num_classes = y_train.shape[1] # 10 0-9



print(img_size)

print(num_channels)

print(num_classes)
def conv_weight_variable(layer_name, shape):

    """ Retrieve an existing variable with the given layer name 

    """

    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())



def fc_weight_variable(layer_name, shape):

    """ Retrieve an existing variable with the given layer name

    """

    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())



def bias_variable(shape):

    """ Creates a new bias variable

    """

    return tf.Variable(tf.constant(0.0, shape=shape))





def conv_layer(input,               # The previous layer

                layer_name,         # Layer name

                num_input_channels, # Num. channels in prev. layer

                filter_size,        # Width and height of each filter

                num_filters,        # Number of filters

                pooling=True):      # Use 2x2 max-pooling



    # Shape of the filter-weights for the convolution

    shape = [filter_size, filter_size, num_input_channels, num_filters]



    # Create new filters with the given shape

    weights = conv_weight_variable(layer_name, shape=shape)

    

    # Create new biases, one for each filter

    biases = bias_variable(shape=[num_filters])



    # Create the TensorFlow operation for convolution

    layer = tf.nn.conv2d(input=input,

                         filter=weights,

                         strides=[1, 1, 1, 1],

                         padding='SAME') # with zero padding



    # Add the biases to the results of the convolution

    layer += biases

    

    # Rectified Linear Unit (RELU)

    layer = tf.nn.relu(layer)



    # Down-sample the image resolution?

    if pooling:

        layer = tf.nn.max_pool(value=layer,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME')



    # Return the resulting layer and the filter-weights

    return layer, weights





def flatten_layer(layer):

    # Get the shape of the input layer.

    layer_shape = layer.get_shape()



    # The number of features is: img_height * img_width * num_channels

    num_features = layer_shape[1:4].num_elements()

    

    # Reshape the layer to [num_images, num_features].

    layer_flat = tf.reshape(layer, [-1, num_features])



    # Return the flattened layer and the number of features.

    return layer_flat, num_features







def fc_layer(input,        # The previous layer

             layer_name,   # The layer name

             num_inputs,   # Num. inputs from prev. layer

             num_outputs,  # Num. outputs

             relu=True):   # Use RELU?



    # Create new weights and biases.

    weights = fc_weight_variable(layer_name, shape=[num_inputs, num_outputs])

    biases = bias_variable(shape=[num_outputs])



    # Calculate the layer activation

    layer = tf.matmul(input, weights) + biases



    # Use ReLU?

    if relu:

        layer = tf.nn.relu(layer)



    return layer



# Convolutional Layer 1.

filter_size1 = 5          # Convolution filters are 5 x 5 pixels.

num_filters1 = 16         # 



# Convolutional Layer 2.

filter_size2 = 5          # Convolution filters are 5 x 5 pixels.

num_filters2 = 32         # 



# Fully-connected layer.

fc_size = 64





x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='x')

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')



y_true_cls = tf.argmax(y_true, dimension=1)

keep_prob = tf.placeholder(tf.float32)



conv_1, w_c1 = conv_layer(input=x,

                          layer_name="conv_1",

                          num_input_channels=num_channels,

                          filter_size=filter_size1,

                          num_filters=num_filters1, pooling=True)



conv_1

conv_2, w_c2 = conv_layer(input=conv_1,

                          layer_name="conv_2",

                          num_input_channels=num_filters1,

                          filter_size=filter_size2,

                          num_filters=num_filters2,

                          pooling=True)



# Apply dropout after the pooling operation

dropout = tf.nn.dropout(conv_2, keep_prob)



dropout





layer_flat, num_features = flatten_layer(dropout)



layer_flat





fc_1 = fc_layer(input=layer_flat,

                layer_name="fc_1",

                num_inputs=num_features,

                num_outputs=fc_size,

                relu=True)



fc_1



fc_2 = fc_layer(input=fc_1,

                layer_name="fc_2",

                num_inputs=fc_size,

                num_outputs=num_classes,

                relu=False)



fc_2



y_pred = tf.nn.softmax(fc_2)



# The class-number is the index of the largest element.

y_pred_cls = tf.argmax(y_pred, dimension=1)



# Calcualte the cross-entropy

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true)



# Take the average of the cross-entropy for all the image classifications.

cost = tf.reduce_mean(cross_entropy)



# Global step is required to compute the decayed learning rate

global_step = tf.Variable(0)



# Apply exponential decay to the learning rate

learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)



# Construct a new Adam optimizer

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)



# Predicted class equals the true class of each image?

correct_prediction = tf.equal(y_pred_cls, y_true_cls)



# Cast predictions to float and calculate the mean

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



session = tf.Session()

session.run(tf.global_variables_initializer())



# Number of training samples in each iteration 

batch_size = 64



# Keep probability in dropout layer

dropout = 0.5



total_iterations = 0



def optimize(num_iterations, display_step):

    

    # Ensure we update the global variable rather than a local copy.

    global total_iterations



    # Start-time used for printing time-usage below.

    start_time = time.time()



    for step in range(num_iterations):



        offset = (step * batch_size) % (y_train.shape[0] - batch_size)

        batch_data = X_train[offset:(offset + batch_size), :, :, :]

        batch_labels = y_train[offset:(offset + batch_size), :]

        

        feed_dict_train = {x: batch_data, y_true: batch_labels, keep_prob: dropout}



        # Run the optimizer using this batch of training data.

        session.run(optimizer, feed_dict=feed_dict_train)



        # Print status every display_step

        if step % display_step == 0:

            

            # Calculate the accuracy on the training-set.

            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

            print("Minibatch accuracy at step %d: %.4f" % (step, batch_acc))

            

            # Calculate the accuracy on the validation-set

            validation_acc = session.run(accuracy, {x: X_val, y_true: y_val, keep_prob: 1.0})

            print("Validation accuracy: %.4f" % validation_acc)



    # Update the total number of iterations performed.

    total_iterations += num_iterations



    # Difference between start and end-times.

    time_diff = time.time() - start_time

    

    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

    

# increase num_iterations to get better accuracy = 5000 to 50000 &  display_step=500

optimize(num_iterations=101, display_step=100)
##predict the test result

#test_pred = session.run(y_pred_cls, {x: X_test, keep_prob: 1.0})

#print(test_pred[:10])



## save results

#np.savetxt('result.csv', np.c_[range(1,len(X_test)+1),test_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
