import cv2

import glob

import numpy as np

import tensorflow as tf

import datetime

import random



DOG = 1

CAT = 0

path_to_train = "data/resize256/"

path_to_log = 'logs/train'

IMAGE_SIZE = 256



def getBatch(step, input_size):

    count = step * 25

    labels = np.zeros((50, 2))

    data = np.zeros((50, input_size, input_size, 3))

    position = 0

    for i in range(count, count + 25):

        cat_file = glob.glob(path_to_train + "cat." + str(i) + ".jpg")

        cat_file = cv2.imread(cat_file[0])

        data[position] = cat_file

        labels[position][CAT] = 1

        position = position + 1



        dog_file = glob.glob(path_to_train + "dog." + str(i) + ".jpg")

        dog_file = cv2.imread(dog_file[0])

        data[position] = dog_file

        labels[position][DOG] = 1

        position = position + 1

    return data, labels



def create_net(input_plh):



    size_1 = int(IMAGE_SIZE / 2)

    size_2 = int(IMAGE_SIZE / 4)

    size_3 = int(IMAGE_SIZE / 8)

    FCL_1_number_of_nodes = 1000

    FCL_2_number_of_nodes = 500

    

    with tf.name_scope("Layer_1"):

        weight = tf.truncated_normal([3, 3, 3, 32], stddev=0.1)

        weight = tf.Variable(weight, name="Weight_L1")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[32]), name="Bias_L1")

        variable_summaries(bias)

        conv_layer = tf.nn.conv2d(input_plh, weight, strides=[1, 1, 1, 1], padding='SAME', name="CNN_L1")

        conv_layer = conv_layer + bias

        conv_layer = tf.nn.relu(conv_layer, name="ReLU_L1")

        pool = tf.nn.max_pool(conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="Max_pool_L1")



        with tf.name_scope('C_1_images_summary'):

            image_shaped_input = tf.reshape(conv_layer, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

            tf.summary.image('c_1_activation_maps', image_shaped_input, 10)



        with tf.name_scope('P_1_images_summary'):

            image_shaped_input = tf.reshape(pool, [-1, 32, 32, 1])

            tf.summary.image('pool_1', image_shaped_input, 10)



    with tf.name_scope("Layer_2"):

        weight = tf.truncated_normal([3, 3, 32, 64], stddev=0.1)

        weight = tf.Variable(weight, name="Weight_L2")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[64]), name="Bias_L2")

        variable_summaries(bias)

        conv_layer = tf.nn.conv2d(pool, weight, strides=[1, 1, 1, 1], padding='SAME', name="CNN_L2")

        conv_layer = conv_layer + bias

        conv_layer = tf.nn.relu(conv_layer, name="ReLU_L2")

        pool = tf.nn.max_pool(conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="Max_pool_L2")



        with tf.name_scope('C_2_images_summary'):



            image_shaped_input = tf.reshape(conv_layer, [-1, size_1, size_1, 1])

            tf.summary.image('c_2_activation_maps', image_shaped_input, 10)



        with tf.name_scope('P_2_images_summary'):

            image_shaped_input = tf.reshape(pool, [-1, size_2, size_2, 1])

            tf.summary.image('pool_2', image_shaped_input, 10)



    with tf.name_scope("Layer_3"):

        weight = tf.truncated_normal([3, 3, 64, 128], stddev=0.1)

        weight = tf.Variable(weight, name="Weight_L3")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[128]), name="Bias_L3")

        variable_summaries(bias)

        conv_layer = tf.nn.conv2d(pool, weight, strides=[1, 1, 1, 1], padding='SAME', name="CNN_L3")

        conv_layer = conv_layer + bias

        conv_layer = tf.nn.relu(conv_layer, name="ReLU_L3")

        pool = tf.nn.max_pool(conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="Max_pool_L3")



        with tf.name_scope('C_3_images_summary'):

            image_shaped_input = tf.reshape(conv_layer, [-1, size_2, size_2, 1])

            tf.summary.image('c_3_activation_maps', image_shaped_input, 10)



        with tf.name_scope('P_3_images_summary'):

            image_shaped_input = tf.reshape(pool, [-1, size_3, size_3, 1])

            tf.summary.image('pool_3', image_shaped_input, 10)



    

    with tf.name_scope("FC_Layer_1"):

        fc1_flat_image_size = size_3 * size_3 * 128

        weight = tf.truncated_normal([fc1_flat_image_size, FCL_1_number_of_nodes])

        weight = tf.Variable(weight, name="Weight_FC_L1")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[FCL_1_number_of_nodes]), name="Bias_FC_L1")

        variable_summaries(bias)

        flat_fc1 = tf.reshape(pool, shape=[-1, fc1_flat_image_size], name="Reshape_FC_L1")

        FC_layer = tf.nn.relu(tf.matmul(flat_fc1, weight) + bias, name="ReLU_FC_L1")



    

    with tf.name_scope("FC_Layer_2"):

        weight = tf.truncated_normal([FCL_1_number_of_nodes, FCL_2_number_of_nodes])

        weight = tf.Variable(weight, name="Weight_FC_L2")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[FCL_2_number_of_nodes]), name="Bias_FC_L2")

        variable_summaries(bias)

        FC_layer = tf.nn.relu(tf.matmul(FC_layer, weight) + bias, name="ReLU_FC_L2")



    with tf.name_scope("FC_Layer_3"):

        weight = tf.truncated_normal([FCL_2_number_of_nodes, 2])

        weight = tf.Variable(weight, name="Weight_FC_L3")

        variable_summaries(weight)

        bias = tf.Variable(tf.constant(0.1, shape=[2]), name="Bias_FC_L3")

        variable_summaries(bias)

        FC_layer = tf.nn.relu(tf.matmul(FC_layer, weight) + bias, name="ReLU_FC_L3")



    return FC_layer



# Thanks to Google for this method!

def variable_summaries(var):

    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):

      mean = tf.reduce_mean(var)

      tf.summary.scalar('mean', mean)

      with tf.name_scope('stddev'):

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

      tf.summary.scalar('stddev', stddev)

      tf.summary.scalar('max', tf.reduce_max(var))

      tf.summary.scalar('min', tf.reduce_min(var))

      tf.summary.histogram('histogram', var)





def main():

    sess = tf.InteractiveSession()



    input_images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name="Input_images")

    truth_labels = tf.placeholder(tf.float32, [None, 2], name="Truth_labels")



    with tf.name_scope('input_images_summary'):

        image_shaped_input = tf.reshape(input_images, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

        tf.summary.image('input_images', image_shaped_input, 10)



    cnn = create_net(input_images)



    with tf.name_scope('cross_entropy'):

        diff = tf.nn.softmax_cross_entropy_with_logits(labels=truth_labels, logits=cnn)

        cross_entropy = tf.reduce_mean(diff)

        tf.summary.scalar('cross_entropy', cross_entropy)



    with tf.name_scope("train_step"):

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



    with tf.name_scope('accuracy'):

        with tf.name_scope('correct_prediction'):

            correct_prediction = tf.equal(tf.argmax(truth_labels, 1), tf.argmax(cnn, 1))

        with tf.name_scope('accuracy'):

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)



    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(path_to_log, sess.graph)



    sess.run(tf.global_variables_initializer())

    start = datetime.datetime.today()

    print("Start time: " + str(start))

    for i in range(499):

        print("Getting data")

        data, labels = getBatch(i, IMAGE_SIZE)

        print("Got data")

        if i % 50 == 0:

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            run_metadata = tf.RunMetadata()

            operations = {"merged": merged, "train_step": train_step, "accuracy": accuracy}

            result_dictionary = sess.run(operations, feed_dict={input_images: data, truth_labels: labels}, options=run_options, run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)

        else:

            operations = {"merged": merged, "train_step": train_step}

            result_dictionary = sess.run(operations, feed_dict={input_images: data, truth_labels: labels})



        train_writer.add_summary(result_dictionary["merged"], i)

        if result_dictionary.get("accuracy", None) is not None:

            print(str(datetime.datetime.today()) + ': Step %d, training accuracy %g' % (i, result_dictionary["accuracy"]))

    train_writer.close()

    finish = datetime.datetime.today()

    print("Finish time: " + str(finish))

    print("Total: " + str(finish-start))





main()
