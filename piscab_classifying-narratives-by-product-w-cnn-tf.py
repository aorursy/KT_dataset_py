import sys

import os

import numpy as np

import pandas as pd

import re

import itertools

import tensorflow as tf

import string

from io import BytesIO

from tensorflow.contrib import learn

from collections import Counter

from time import time

import datetime

from sklearn import model_selection

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder
# Read the input dataset 

d = pd.read_csv("../input/consumer_complaints.csv", 

                usecols=('product','consumer_complaint_narrative'),

                dtype={'consumer_complaint_narrative': object})

# Only interested in data with consumer complaints

d=d[d['consumer_complaint_narrative'].notnull()]

d=d[d['product'].notnull()]

d.reset_index(drop=True,inplace=True)
# Let's see what's in the data 

print ("Data dimensions:", d.shape)

print (d.head())



# Let's see a table of how many examples we have of each product

print ("\nList of Products       Occurrences\n")

print (d["product"].value_counts())
def clean_str(string):

    """

    Tokenization/string cleaning (partially modified)

    """

    string = re.sub(r"[^A-Za-z0-9()!?\'\`%$]", " ", string) # keep also %$ but removed comma

    string = re.sub(r"\'s", " \'s", string)

    string = re.sub(r"\'ve", " \'ve", string)

    string = re.sub(r"n\'t", " n\'t", string)

    string = re.sub(r"\'re", " \'re", string)

    string = re.sub(r"\'d", " \'d", string)

    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r"!", " ! ", string)

    string = re.sub(r"\(", " ( ", string)

    string = re.sub(r"\)", " ) ", string)

    string = re.sub(r"\?", " ? ", string)

    string = re.sub(r"\$", " $ ", string) #yes, isolate $

    string = re.sub(r"\%", " % ", string) #yes, isolate %

    string = re.sub(r"\s{2,}", " ", string)

    

    # fixing XXX and xxx like as word

    string = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx",string)

    # removing non ascii

    string = re.sub(r'[^\x00-\x7F]+', "", string) 

    

    return string.strip().lower()
word_data=[]

t0 = time()



for message in d['consumer_complaint_narrative']:

    word_data.append(clean_str(message))



# With a MacBook Pro (Late 2011)

# 2.4 GHz Intel Core i5, 4 GB 1333 MHz DDR3

print ("\nCleaning time: mine = 41.8 s, here =", round(time()-t0, 1), "s")
# Have a look before and after cleaning texts

an_example = 38

print ("Note: the reference product is",d ['product'][an_example])

print ("\n** Before cleaning ** \n")

print (d['consumer_complaint_narrative'][an_example])

print ("** After cleaning ** \n")

print (word_data [an_example])
max_document_length = max([len(x.split(" ")) for x in word_data])

print ("Max_document_length:",max_document_length)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

num_data = np.array(list(vocab_processor.fit_transform(word_data)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# Check data "lengths"

print ("Check my variables:")

print ("\n* word_data length:", len(word_data))

print ("* num_data length: ", len(num_data)) # once words are numbers



#Create the list of products

product_labels = list(set(d['product']))

print ("\nProducts:")

print ("* data length: ",len(product_labels))

print ("* labels:\n",product_labels)
np.random.seed(57)

shuffle_indices = np.random.permutation(np.arange(len(num_data)))

x_shuffled = num_data[shuffle_indices]

y_shuffled = d['product'][shuffle_indices]

print ("* x shuffled:", x_shuffled.shape)

print ("* y shuffled:", y_shuffled.shape)
features_dummy, x_test, labels_dummy, test_labels = model_selection.train_test_split(x_shuffled, y_shuffled, test_size=0.20, random_state= 23)

x_train, x_valid, train_labels, valid_labels = model_selection.train_test_split(features_dummy, labels_dummy, test_size=0.25, random_state= 34)



print('Training set  ',   x_train.shape, train_labels.shape)

print('Validation set',   x_valid.shape, valid_labels.shape)

print('Test set      ',    x_test.shape,  test_labels.shape)



# free some memory

del num_data, d 

del x_shuffled, y_shuffled, labels_dummy, features_dummy
def batch_iter(data, batch_size, num_epochs, shuffle=True):

    """

    Generates a batch iterator for a dataset.

    """

    data = np.array(data)

    data_size = len(data)

    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):

        # Shuffle the data at each epoch

        if shuffle:

            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_data = data[shuffle_indices]

        else:

            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):

            start_index = batch_num * batch_size

            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]
class TextCNN(object):

    """

    A CNN for text classification.

    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    """

    def __init__(

      self, sequence_length, num_classes, vocab_size,

      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):



        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")



        # Keeping track of l2 regularization loss (optional)

        l2_loss = tf.constant(0.0)



        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            W = tf.Variable(

                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),

                name="W")

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



        # Create a convolution + maxpool layer for each filter size

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):

            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer

                filter_shape = [filter_size, embedding_size, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(

                    self.embedded_chars_expanded,

                    W,

                    strides=[1, 1, 1, 1],

                    padding="VALID",

                    name="conv")

                # Apply nonlinearity

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs

                pooled = tf.nn.max_pool(

                    h,

                    ksize=[1, sequence_length - filter_size + 1, 1, 1],

                    strides=[1, 1, 1, 1], 

                    padding='VALID',

                    name="pool")

                pooled_outputs.append(pooled)



        # Combine all the pooled features

        num_filters_total = num_filters * len(filter_sizes)

        self.h_pool = tf.concat(3, pooled_outputs)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])



        # Add dropout

        with tf.name_scope("dropout"):

            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)



        # Final (unnormalized) scores and predictions

        with tf.name_scope("output"):

            W = tf.get_variable(

                "W",

                shape=[num_filters_total, num_classes],

                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)

            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") 

            self.predictions = tf.argmax(self.scores, 1, name="predictions")



        # CalculateMean cross-entropy loss

        with tf.name_scope("loss"):

            print (self.scores)

            print (self.input_y)

            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss



        # Accuracy

        with tf.name_scope("accuracy"):

            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
# Model Hyperparameters

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")



# WAS: tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")

tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '2,3,4')")

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")



# Training parameters

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (best: 8)") # was 200

tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")

tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS

FLAGS._parse_flags()

print("\nParameters:")

for attr, value in sorted(FLAGS.__flags.items()):

    print("{}={}".format(attr.upper(), value))

print("")
def oneHot(dummy_labels):

    le = preprocessing.LabelEncoder()

    enc = OneHotEncoder()

    

    le.fit (dummy_labels)

    y_dummy = le.fit_transform(dummy_labels)

    y_dummy = y_dummy.reshape(-1, 1)

    enc.fit(y_dummy)

    y_dummy = enc.transform(y_dummy).toarray()

    y_dummy = y_dummy.astype('float32')

    print ("\n * OneHot example")

    print (y_dummy)

    return (y_dummy)

        

y_train = oneHot(train_labels)

y_valid = oneHot(valid_labels)

y_test  = oneHot( test_labels)
# Training

# ==================================================



with tf.Graph().as_default():

    session_conf = tf.ConfigProto(

      allow_soft_placement=FLAGS.allow_soft_placement,

      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)

    with sess.as_default():

        cnn = TextCNN(

            sequence_length=x_train.shape[1],

            num_classes=len(product_labels),

            vocab_size=len(vocab_processor.vocabulary_),

            embedding_size=FLAGS.embedding_dim,

            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),

            num_filters=FLAGS.num_filters,

            l2_reg_lambda=FLAGS.l2_reg_lambda)



        # Define Training procedure

        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(1e-3)

        grads_and_vars = optimizer.compute_gradients(cnn.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



        # Keep track of gradient values and sparsity (optional)

        grad_summaries = []

        for g, v in grads_and_vars:

            if g is not None:

                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)

                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

                grad_summaries.append(grad_hist_summary)

                grad_summaries.append(sparsity_summary)

        grad_summaries_merged = tf.merge_summary(grad_summaries)



        # Output directory for models and summaries (if needed)

        

        #timestamp = str(int(time()))

        #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        #print("Writing to {}\n".format(out_dir))



        # Summaries for loss and accuracy

        loss_summary = tf.scalar_summary("loss", cnn.loss)

        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)



        # Train Summaries

        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])

        #train_summary_dir = os.path.join(out_dir, "summaries", "train")

        #train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)



        # Dev summaries

        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])

        #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

        #dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)



        # Checkpoint directory (if needed)

        # Tensorflow assumes this directory already exists so we need to create it

        

        #checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

        #checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        #if not os.path.exists(checkpoint_dir):

        #    os.makedirs(checkpoint_dir)

        #saver = tf.train.Saver(tf.all_variables())



        # Write vocabulary (if needed)

        #vocab_processor.save(os.path.join(out_dir, "vocab"))



        # Initialize all variables

        sess.run(tf.initialize_all_variables())



        def train_step(x_batch, y_batch):

            """

            A single training step

            """

            feed_dict = {

              cnn.input_x: x_batch,

              cnn.input_y: y_batch,

              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob

            }

            _, step, summaries, loss, accuracy = sess.run(

                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],

                feed_dict)

            time_str = datetime.datetime.now().isoformat()

            

            # Uncomment next print if interested in batch results 

            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            

            #train_summary_writer.add_summary(summaries, step)



        def dev_step(x_batch, y_batch, writer=None):

            """

            Evaluates model on a dev set

            """

            feed_dict = {

              cnn.input_x: x_batch,

              cnn.input_y: y_batch,

              cnn.dropout_keep_prob: 1.0

            }

            step, summaries, loss, accuracy = sess.run(

                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],

                feed_dict)

            return loss, accuracy, summaries



        # Generate batches

        batches = batch_iter(

            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        

        # Training loop. For each batch...

        for batch in batches:

            x_batch, y_batch = zip(*batch)

            train_step(x_batch, y_batch)

            current_step = tf.train.global_step(sess, global_step)

            

# Validating

# ==================================================

            if current_step % FLAGS.evaluate_every == 0:

                #print("\nEvaluation:")

                

                # Generate batches

                batches_valid = batch_iter(

                    list(zip(x_valid, y_valid)), FLAGS.batch_size, 1)

                

                loss_valid = 0.

                acc_valid = 0.

                len_batches = 0.

                

                for batch_valid in batches_valid:  

                    

                    x_batch_valid, y_batch_valid = zip(*batch_valid)

                    #aLoss, anAcc, aSummary = dev_step(x_batch_valid, y_batch_valid, writer=dev_summary_writer)

                    aLoss, anAcc, aSummary = dev_step(x_batch_valid, y_batch_valid)

                    loss_valid += aLoss 

                    acc_valid  += anAcc

                    len_batches += 1.

                

                loss_valid = loss_valid / len_batches

                acc_valid  = acc_valid  / len_batches 

                time_str = datetime.datetime.now().isoformat()

                print("Validation set: {}, step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss_valid, acc_valid))

                #dev_summary_writer.add_summary(aSummary, current_step)

                #print("")

                

            #if current_step % FLAGS.checkpoint_every == 0:

            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)

            #    print("Saved model checkpoint to {}\n".format(path))

    

        

# Testing

# ==================================================

        if True:

            print("\n\nTest set:")

            

            # Generate batches

            batches_test = batch_iter(

                list(zip(x_test, y_test)), FLAGS.batch_size, 1)

        

            loss_test = 0.

            acc_test  = 0.

            len_batches = 0.

            

            for batch_test in batches_test:  

                    

                    x_batch_test, y_batch_test = zip(*batch_test)

                    #aLoss, anAcc, aSummary = dev_step(x_batch_test, y_batch_test, writer=dev_summary_writer)

                    aLoss, anAcc, aSummary = dev_step(x_batch_test, y_batch_test)

                    loss_test += aLoss 

                    acc_test  += anAcc

                    len_batches += 1.

                

            loss_test = loss_test / len_batches

            acc_test  = acc_test  / len_batches 

            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss_test, acc_test))

            #dev_summary_writer.add_summary(aSummary, current_step)

            print("")