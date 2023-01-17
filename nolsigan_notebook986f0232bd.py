import tensorflow as tf

import numpy as np

import csv

import os

import math



%matplotlib inline

import matplotlib.pyplot as plt
with open('../data/mnist/ext_train_manipulated.csv', newline='') as csvfile:

    csv_file_object = csv.reader(csvfile, dialect='excel')

    header = csv_file_object.__next__()

    

    data=[]

    for row in csv_file_object:

        data.append(row)

    data = np.array(data)

    data = data.astype(np.float)

    

print(data.shape)
# Define constants

NUM_CLASSES = 10

INPUT_SIZE = 784



BATCH_SIZE = 100

EVAL_BATCH_SIZE = 1



HIDDEN1_UNITS = 512



MAX_EPOCHS = 10



CONV1_STRIDES = [1, 1, 1, 1]

CONV2_STRIDES = [1, 1, 1, 1]



TRAIN_DIR = "./checkpoints"
# Inference Graph

def mnist_inference(image, hidden1_units, hidden2_units):

    """

    Args:

        image: Image placeholder

        hidden1_units: Size of the first hidden layer

        hidden2_units: Size of the second hidden layer

    Returns:

        logits: Output tensor with the computed logits

    """

    # First Conv Layer

    with tf.name_scope('conv1'):

        filters = tf.Variable(

                tf.truncated_normal([5, 5, 1, 32],

                                    stddev=1.0 / math.sqrt(float(5 * 5 * 32))),

                name='filters')

        biases = tf.Variable(tf.zeros([32]),

                             name='biases')

        conv = tf.nn.conv2d(image, filters, strides=CONV1_STRIDES, padding='SAME')

        conv1 = tf.nn.relu(conv + biases)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second Conv Layer

    with tf.name_scope('conv2'):

        filters = tf.Variable(

                tf.truncated_normal([5, 5, 32, 64],

                                    stddev=1.0 / math.sqrt(float(5 * 5 * 64))),

                name='filters')

        biases = tf.Variable(tf.zeros([64]),

                            name='biases')

        conv = tf.nn.conv2d(pool1, filters, strides=CONV2_STRIDES, padding='SAME')

        conv2 = tf.nn.relu(conv + biases)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Hidden 1

    with tf.name_scope('hidden1'):

        weights = tf.Variable(

                tf.truncated_normal([7 * 7 * 64, hidden1_units],

                                    stddev=1.0 / math.sqrt(float(7 * 7 * 64))),

                name='weights')

        biases = tf.Variable(tf.zeros([hidden1_units]),

                             name='biases')

        pool2_flat = tf.reshape(pool2, (-1, 7 * 7 * 64))

        hidden1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Linear

    with tf.name_scope('softmax_linear'):

        weights = tf.Variable(

                tf.truncated_normal([hidden1_units, NUM_CLASSES],

                                    stddev=1.0 / math.sqrt(float(hidden1_units))),

                name='weights')

        biases = tf.Variable(tf.zeros([NUM_CLASSES]),

                             name='biases')

        logits = tf.matmul(hidden1, weights) + biases

        

    return logits
# Build training graph

def mnist_training(logits, labels, learning_rate):

    """

    Args:

        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES]

        labels: Labels tensor, int32 - [BATCH_SIZE]

        learning_rate: The learning rate to use for gradient descent

    Returns:

        train_op: The Op for training

        loss: The Op for calculating loss

    """

    labels = tf.to_int64(labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(

        logits, labels, name='xentropy')

    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)

    

    return train_op, loss
# Build complete graph

mnist_graph = tf.Graph()

with mnist_graph.as_default():

    # Generate placeholders for the images and labels

    inputs_placeholder = tf.placeholder(tf.float32)

    labels_placeholder = tf.placeholder(tf.int32)

    tf.add_to_collection("inputs", inputs_placeholder)

    tf.add_to_collection("labels", labels_placeholder)

    

    # Build a Graph that computes predictions from the inference model

    logits = mnist_inference(inputs_placeholder,

                             HIDDEN1_UNITS,

                             1)

    tf.add_to_collection("logits", logits)

    

    # Add to graph the Ops that calculate and apply gradients

    train_op, loss = mnist_training(logits, labels_placeholder, 0.03)

    

    # Add the variable initializer Op

    init = tf.initialize_all_variables()

    

    # Create a saver for writing training checkpoints

    saver = tf.train.Saver()
# Run training for MAX_STEPS and save checkpoint at the end

with tf.Session(graph=mnist_graph) as sess:

    # Run the Op to initialize variables

    sess.run(init)

    

    # Start the training loop

    losses = []

    for step in range(MAX_EPOCHS):

        # Generate batchs of image

        np.random.shuffle(data)

        remainder = len(data) % BATCH_SIZE

        if remainder != 0:

            epoch = data[:-(len(data) % BATCH_SIZE)]

        else:

            epoch = data

        batchs = np.split(epoch, len(epoch) / BATCH_SIZE)

        batchs_num = len(batchs)

        

        batch_loss = 0.

        print('Itertate epoch : ', step)

        for batch in batchs:

            images_feed = batch[0::, 1:]

            labels_feed = batch[0::, 0].astype(np.int)

            images_feed = np.array(images_feed).reshape((-1, 28, 28, 1))

            labels_feed = np.array(labels_feed)

        

            _, loss_value = sess.run([train_op, loss],

                                     feed_dict={inputs_placeholder: images_feed,

                                                labels_placeholder: labels_feed})

            

            batch_loss += loss_value / batchs_num

            losses.append(loss_value)

            

        print('Epoch %d: loss = %.2f' % (step, batch_loss))

        

    # Write a checkpoint

    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')

    saver.save(sess, checkpoint_file, global_step=step)

    plt.plot(losses)
with open('../data/mnist/test_manipulated.csv', newline='') as csvfile:

    csv_file_object = csv.reader(csvfile, dialect='excel')

    haeder = csv_file_object.__next__()

    

    test=[]

    for row in csv_file_object:

        test.append(row)

    test = np.array(test)

    test = test.astype(np.float)



print(test.shape)
with tf.Session(graph=tf.Graph()) as sess:

    saver = tf.train.import_meta_graph(

        os.path.join(TRAIN_DIR, 'checkpoint-9.meta'))

    saver.restore(

        sess, os.path.join(TRAIN_DIR, 'checkpoint-9'))

    

    # Retrieve Ops

    logits = tf.get_collection('logits')[0]

    inputs_placeholder = tf.get_collection('inputs')[0]

    labels_placeholder = tf.get_collection('labels')[0]

    

    eval_op = tf.nn.top_k(logits)

    

    # Run evaluation

    labels_feed = test[0::, 0]

    inputs_feed = test.reshape((-1, 28, 28, 1))

    labels_feed = np.array(labels_feed)

    imgplot = plt.imshow(np.reshape(inputs_feed[0], (28, 28)))

    

    prediction = sess.run(eval_op,

                          feed_dict={inputs_placeholder: inputs_feed,

                                    labels_placeholder: labels_feed})
with open('conv2d.csv', 'w', newline='') as csvfile:

    csv_file_object = csv.writer(csvfile, dialect='excel')

    csv_file_object.writerow(['ImageId', 'Label'])

    for i in range(len(test)):

        csv_file_object.writerow([i+1, prediction.indices[i][0]])