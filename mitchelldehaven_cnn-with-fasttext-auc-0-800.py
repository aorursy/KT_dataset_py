import numpy as np

import pandas as pd

import os

import time

import tensorflow as tf

from keras.preprocessing import text, sequence

from nltk import word_tokenize

from sklearn.metrics import roc_auc_score

from IPython.display import clear_output
MAX_LENGTH = 200

BATCH_SIZE = 64

EPOCHS = 10

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

two_hunnid_k = False

train_file = "reddit_200k_train.csv" if two_hunnid_k else "reddit_train.csv"

test_file = "reddit_200k_test.csv" if two_hunnid_k else "reddit_test.csv"

x_col = "body" if two_hunnid_k else "BODY"

y_col = "REMOVED"
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype="float32")



def load_embeddings(file):

    with open(file) as f:

        return dict(get_coefs(*line.strip().split(" ")) for line in f)



def build_matrix(word_index, file):

    embedding_index = load_embeddings(file)

    embedding_matrix = np.zeros((len(word_index)+1, 300), dtype=np.float32)

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            continue

    return embedding_matrix
train_df = pd.read_csv("../input/rscience-popular-comment-removal/" + train_file, encoding = "ISO-8859-1")

test_df = pd.read_csv("../input/rscience-popular-comment-removal/" + test_file, encoding = "ISO-8859-1")

x_train = train_df[x_col].values

y_train = train_df[y_col].values.astype(np.int32)

onehot = np.zeros((y_train.size, y_train.max()+1))

onehot[np.arange(y_train.size), y_train] = 1

y_train = onehot

x_test = test_df[x_col].values

y_test = test_df[y_col].values.astype(np.int32)

onehot = np.zeros((y_test.size, y_test.max()+1))

onehot[np.arange(y_test.size), y_test] = 1

y_test = onehot



del train_df, test_df
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

# On one hand, including text from the test set in the 

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)
embedding_matrix = build_matrix(tokenizer.word_index, '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')
val_size = 10000 if two_hunnid_k else 2000

shuffled_indices = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)

x_validation = x_train[shuffled_indices[-val_size:]]

y_validation = y_train[shuffled_indices[-val_size:]]

x_train = x_train[shuffled_indices[:-val_size]]

y_train = y_train[shuffled_indices[:-val_size]]
tf.reset_default_graph()

inputs = tf.placeholder(shape=(None, MAX_LENGTH), dtype=tf.int32)

keep_prob = tf.placeholder_with_default(1.0, shape=())

keep_prob_input = tf.placeholder_with_default(1.0, shape=())

keep_prob_conv = tf.placeholder_with_default(1.0, shape=())

targets = tf.placeholder(shape=(None, 2), dtype=tf.float32)



embedding_layer = tf.Variable(embedding_matrix, trainable=False, name="embedding_layer")

input_embeddings = tf.nn.embedding_lookup(embedding_layer, inputs)

expanded_inputs = tf.expand_dims(input_embeddings, 3)

expanded_inputs = tf.nn.dropout(expanded_inputs, keep_prob=keep_prob)



conv_layer2 = tf.layers.conv2d(expanded_inputs, 16, (2, 300), (1, 1), activation=tf.nn.relu)

conv_layer2 = tf.nn.dropout(conv_layer2, keep_prob=keep_prob_conv)

conv_layer4 = tf.layers.conv2d(expanded_inputs, 16, (4, 300), (1, 1), activation=tf.nn.relu)

conv_layer4 = tf.nn.dropout(conv_layer4, keep_prob=keep_prob_conv)

conv_layer6 = tf.layers.conv2d(expanded_inputs, 16, (6, 300), (1, 1), activation=tf.nn.relu)

conv_layer6 = tf.nn.dropout(conv_layer6, keep_prob=keep_prob_conv)

conv_layer8 = tf.layers.conv2d(expanded_inputs, 16, (8, 300), (1, 1), activation=tf.nn.relu)

conv_layer8 = tf.nn.dropout(conv_layer8, keep_prob=keep_prob_conv)

squeeze2 = tf.squeeze(conv_layer2, 2)

squeeze4 = tf.squeeze(conv_layer4, 2)

squeeze6 = tf.squeeze(conv_layer6, 2)

squeeze8 = tf.squeeze(conv_layer8, 2)

pool2 = tf.layers.max_pooling1d(squeeze2, MAX_LENGTH-2+1, 1)

pool4 = tf.layers.max_pooling1d(squeeze4, MAX_LENGTH-4+1, 1)

pool6 = tf.layers.max_pooling1d(squeeze6, MAX_LENGTH-6+1, 1)

pool8 = tf.layers.max_pooling1d(squeeze8, MAX_LENGTH-8+1, 1)

pools = [pool2, pool4, pool6, pool8]

pools = [tf.squeeze(x, 1) for x in pools]

concat_layers = tf.concat(pools, axis=1)

hidden_layer = tf.layers.dense(concat_layers, 256, activation=tf.nn.relu)

final_layer = tf.layers.dense(hidden_layer, 2, activation=tf.nn.softmax)



accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(targets, 1), tf.math.argmax(final_layer,1)), tf.float32))

optimizer = tf.train.AdamOptimizer(0.001)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=final_layer))

var = tf.trainable_variables() 

lossl2 = tf.add_n([ tf.nn.l2_loss(v) for v in var

                    if 'bias' not in v.name and "embedding" not in v.name]) * 0.003



update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

training_op = optimizer.minimize(loss + lossl2)

training_op = tf.group([training_op, update_ops])
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):

        avg_train_error = []

        avg_train_acc = []

        train_acc = 0

        t0 = time.time()

        for i in range((len(x_train) // BATCH_SIZE)):

            x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            feed = {inputs: x_batch, keep_prob: 0.5, keep_prob_input: 0.9, keep_prob_conv: 0.5, targets: y_batch}

            _, train_loss, train_acc = sess.run([training_op, loss, accuracy], feed)

            avg_train_error.append(train_loss)

            avg_train_acc.append(train_acc)



        clear_output(wait=True)

        feed = {inputs:x_validation, targets: y_validation}

        validation_err, validation_acc = sess.run([loss, accuracy], feed)            

        shuffled_indices = np.random.choice(x_train.shape[0], x_train.shape[0], False)

        x_train = x_train[shuffled_indices]

        y_train = y_train[shuffled_indices]

        print("Epoch: " + str(epoch + 1))

        print("Average error: {:.5f}".format(sum(avg_train_error)/len(avg_train_error)))

        print("Average accuracy: {:.5f}".format(sum(avg_train_acc)/len(avg_train_acc)))

        print("Validation error: {:.5f}".format(validation_err))

        print("Validation accuracy: {:.5f}".format(validation_acc))



    clear_output(wait=True)

    feed = {inputs: x_test, targets: y_test}

    test_loss, test_acc, pred = sess.run([loss, accuracy, final_layer], feed)

    auc_score = roc_auc_score(y_test[:,1], pred[:,1])

    print("Test error: {:.5f}".format(test_loss))

    print("Test accuray: {:.5f}".format(test_acc))

    print("AUC score: {:.5f}".format(auc_score))    