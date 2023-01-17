import time

import numpy as np # linear algebra

import tensorflow as tf # for a neural network...?





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["dir", "..\input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Constants that will probably help

N_CLASSES = 10

N_FEATURES = 28*28

BATCH_SIZE = 50

MIN_AFTER = 10000

CAPACITY = MIN_AFTER + (3*BATCH_SIZE)
# Reset the graph

tf.reset_default_graph()
# Reading in data as an input pipeline

# 1) List of file names (not going to do file shuffling or put an epoch limit)

#    a) going to put in both the names for the training and test data

file_names = ["../input/train.csv"]#, "../input/test.csv"]



# 2) Filename queue (this is for passing in the files)

#    a) a queue can be made from tf.train.string_input_producer

file_queue = tf.train.string_input_producer(file_names, num_epochs=1)





# 3) Creating a reader for the specific file format

#    a) the tf.TextLineReader will output the lines in a file!

#    b) have it read one file to pass into the decoder

#        i) I think key = name, value = file information?

file_reader = tf.TextLineReader(skip_header_lines=1, name="Training")

key, value = file_reader.read(file_queue)





# 4) Creating a decoder to grab the information

#    a) use the csv decoder since the file is a csv file

#    b) returns multidimensional array with each feature making up a column

#    c) requires record_defaults: a Tensor of #features size, with a scalar

#        representing the datatype of that feature

#    d) separate the features and labels

record_defaults = [[1]]*(N_FEATURES+1) # +1 for the labels column

all_data = tf.decode_csv(value, record_defaults=record_defaults)



outputs = all_data[0]

features = tf.stack(all_data[1:])



# 5) Creating batches from the information

feature_batch, output_batch = tf.train.shuffle_batch([features,outputs],

                                                   BATCH_SIZE,

                                                   CAPACITY,

                                                   MIN_AFTER)
# # Now preparing the softmax neural network - gives 0.892 accuracy

# # 1) Creating placeholders for the data

# X = tf.placeholder(tf.float32, shape=[None, N_FEATURES])

# Y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])



# #2) Create softmax variables to manipulate the data

# weights = tf.Variable(tf.zeros([N_FEATURES, N_CLASSES]))

# biases = tf.Variable(tf.zeros([N_CLASSES]))



# # 3) Going to do class predictions and loss function

# #    a) for now, the prediction function is x*weights + biases

# #    b) the loss is the mean of the predictions that were wrong

# #        i) logits are the predictions

# Yhat = tf.matmul(X, weights) + biases



# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

    # labels=Y, logits=Yhat))



# # 4) Create a way to train the model, using the GradientDescentOptimizer

# #    a) can also use AdadeltaOptimizer, an adaptive learning optimizer

# training_step = tf.train.GradientDescentOptimizer(

    # learning_rate=0.5).minimize(cross_entropy)



# # 5) Evaluate the model

# correct_prediction = tf.equal(tf.argmax(Yhat,1), tf.argmax(Y,1))

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Now preparing for a Convolutional Network - gives 0.96 accuracy!

# 0) Creating placeholders for the data

X = tf.placeholder(tf.float32, shape=[None, N_FEATURES])

Y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])



# 1) Creating a weight-initialization function

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



# 2) Creating a bias-initialization function

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



# 3) Creating the convolution filter

def convolution_2d(X, weights, strides=[1,1,1,1], padding="SAME"):

    # for now, don't do anything

    return tf.nn.conv2d(X, weights, strides=strides, padding=padding)



# 4) Creating the pooling operation

def max_pool_2x2(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 

                 padding='SAME'):

    return tf.nn.max_pool(X, ksize=ksize, strides=strides, 

                          padding=padding)



# 5) Creating the actual convolutional layers

W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])

image = tf.reshape(X, [-1,28,28,1])

h_conv1 = tf.nn.relu(convolution_2d(image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)



# 6) Create the fully connected layer

W_fullc = weight_variable([7 * 7 * 64, 1024])

b_fullc = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fullc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fullc) + b_fullc)



# 7) Create something that can do dropout to prevent overfitting

prob_keep = tf.placeholder(tf.float32)

h_fullc_drop = tf.nn.dropout(h_fullc, prob_keep)



# 8) Create the output layer and calculate cross extropy

W_out = weight_variable([1024, 10])

b_out = bias_variable([10])

Yhat = tf.matmul(h_fullc_drop, W_out) + b_out



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

        labels=Y, logits=Yhat))



# 9) Create a training optimizer - using Adam

training_step = tf.train.AdamOptimizer(

    learning_rate=1e-4).minimize(cross_entropy)



# 10) Evaluate the model

correct_prediction = tf.equal(tf.argmax(Yhat,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Add ops to save and restore all the variables.

saver = tf.train.Saver()
save_path = ''

# Running the session!

with tf.Session() as sess:

    begin = time.time()

    sess.run(tf.global_variables_initializer())

    after_global_initialize = time.time()

    sess.run(tf.local_variables_initializer())

    after_local_initialize = time.time()

    

    # 1) Populate the file_queue

    #    a) requires you to initialize a Coordinator

    coord = tf.train.Coordinator()

    after_coord_initialize = time.time()

    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    after_thread_initialize = time.time()

    

    validation_x = []

    validation_y = []

    

    # getting a bunch of validation data

    # the 126 is just a number I picked; can be pretty much anything

    # 126 = 840*0.15, where 840 is the number of batches

    #validation_time_counter = time.time()

    for i in range(126):

        validation_batch = sess.run([feature_batch, output_batch])

        validation_x.extend(validation_batch[0])

        validation_y.extend(validation_batch[1])

        #print("Validation", i, time.time() - validation_time_counter)

        #validation_time_counter = time.time()

    after_create_validation = time.time()

    counter = 0

    #training_time_counter = time.time()

    try:

        while not coord.should_stop():

        # example = one observation; label = the label

            example_batch, label_batch = sess.run([feature_batch, 

                                                   output_batch])

            feed_dict = {X:example_batch, Y:tf.one_hot(label_batch,

                                                       depth=N_CLASSES).eval(),

                         prob_keep:0.5}

            sess.run(training_step, feed_dict=feed_dict)

            counter += 1

            #print("Training", counter, time.time() - training_time_counter)

            #training_time_counter = time.time()

    except tf.errors.OutOfRangeError:

        print("Finished training!")

    finally:

        # close the Coordinate since we don't need it anymore

        coord.request_stop()

    after_training = time.time()

    print(counter)

    

    print("Validation accuracy:", 

          accuracy.eval(feed_dict={X:validation_batch[0], 

                                   Y:tf.one_hot(validation_batch[1], 

                                                depth=N_CLASSES).eval(),

                                   prob_keep:1.0}))

    after_evaluation = time.time()

    

    save_path = saver.save(sess, "my-model")

    print("Model saved in file: %s" % save_path)

    

    print("Time for Global Initalize:", after_global_initialize - begin)

    print("Time for Local Initalize:", after_local_initialize - after_global_initialize)

    print("Time for Coord Initalize:", after_coord_initialize - after_local_initialize)

    print("Time for Thread Initalize:", after_thread_initialize - after_coord_initialize)

    print("Time for Creating Validation:", after_create_validation - after_thread_initialize)

    print("Time for Training:", after_training - after_create_validation)

    print("Time for Evaluation:", after_evaluation - after_training)

    coord.join(threads)
# Want to get output for the test data - need to repeat some things!

file_queue_test = tf.train.string_input_producer(["../input/test.csv"], 

                                                 num_epochs=1,

                                                 name="Test")



file_reader_test = tf.TextLineReader(skip_header_lines=1, name="Testing")

key_test, value_test = file_reader_test.read(file_queue_test)



record_defaults_test = [[1]]*(N_FEATURES)

all_data_test = tf.decode_csv(value_test,

                              record_defaults=record_defaults_test)



X_test = tf.placeholder(tf.float32, shape=[None, N_FEATURES])



# Calculate prediction step

image_test = tf.reshape(X_test, [-1,28,28,1])

h_conv1_test = tf.nn.relu(convolution_2d(image_test, W_conv1) + b_conv1)

h_pool1_test = max_pool_2x2(h_conv1_test)

h_conv2_test = tf.nn.relu(convolution_2d(h_pool1_test,W_conv2)+b_conv2)

h_pool2_test = max_pool_2x2(h_conv2_test)

h_pool2_flat_test = tf.reshape(h_pool2_test, [-1, 7*7*64])

h_fullc_test = tf.nn.relu(tf.matmul(h_pool2_flat_test, W_fullc) +b_fullc)

Ypred = tf.matmul(h_fullc_test, W_out) + b_out
predictions = []

with tf.Session() as sess:

    saver.restore(sess, "model")

    print("Model restored.")

    

    sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())

    

    # 1) Populate the file_queue

    #    a) requires you to initialize a Coordinator

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    

    test_data = []

    counter = 0

    try:

        while not coord.should_stop():

        # example = one observation; label = the label

            test_data.append(sess.run(all_data_test))

            counter += 1

            #print("Training", counter, time.time() - training_time_counter)

            #training_time_counter = time.time()

    except tf.errors.OutOfRangeError:

        print("Finished training!", counter)

    finally:

        # close the Coordinate since we don't need it anymore

        coord.request_stop()

        

        

    predictions = sess.run(Ypred, feed_dict={X_test:test_data})

    

    with open("prediction.txt", 'w') as file:

        file.write(predictions)
print(predictions)