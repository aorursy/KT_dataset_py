import numpy as np

import tensorflow as tf

import pandas as pd
feature_sets_train = pd.read_csv('../input/train.csv')

# TODO: Use both datasets to make the embeddings (vocab_to_int map)

feature_sets_test = pd.read_csv('../input/test.csv')

feature_sets_train_tests = pd.concat([feature_sets_train, feature_sets_test])

feature_sets = feature_sets_train
passengers = [' '.join(map(str,passenger[[2,3,4,5,8,9,10,11]])) for passenger in feature_sets.values]

passengers_test = [' '.join(map(str,passenger[[1,2,3,4,7,8,9,10]])) for passenger in feature_sets_test.values]



survived = [passenger[1] for passenger in feature_sets.values]

feature_sets = passengers

feature_sets_test = passengers_test

labels =  survived
feature_sets_train
feature_sets_train_tests
#from string import punctuation

#all_text = ''.join([c for c in feature_sets if c not in punctuation])

#feature_sets = all_text.split(',')



passengers = [' '.join(map(str,passenger[[0,1,2,3,4,5,7,8,9,11]])) for passenger in feature_sets_train_tests.values]



all_text = ' '.join(passengers)

words = all_text.split()
all_text[:1000]
words[:10]
from collections import Counter

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



feature_sets_ints = []

feature_sets_ints_test = []

print(feature_sets[0])

for each in feature_sets:

    feature_sets_ints.append([vocab_to_int[word] for word in each.split()])



print(feature_sets_test[0])

for each in feature_sets_test:

    feature_sets_ints_test.append([vocab_to_int[word] for word in each.split()])   
feature_set_lens = Counter([len(x) for x in feature_sets_ints])

print("Zero-length feature_sets: {}".format(feature_set_lens[0]))

print("Maximum feature_set length: {}".format(max(feature_set_lens)))
non_zero_idx = [ii for ii, feature_set in enumerate(feature_sets_ints) if len(feature_set) != 0]

print(len(non_zero_idx))



non_zero_idx_test = [ii for ii, feature_set in enumerate(feature_sets_ints_test) if len(feature_set) != 0]

print(len(non_zero_idx_test))
print(feature_sets_ints[-1])

print(feature_sets_ints_test[-1])
feature_sets_ints = [feature_sets_ints[ii] for ii in non_zero_idx]

feature_sets_ints_test = [feature_sets_ints_test[ii] for ii in non_zero_idx_test]



labels = np.array([labels[ii] for ii in non_zero_idx])
seq_len = 24

features = np.zeros((len(feature_sets_ints), seq_len), dtype=int)

for i, row in enumerate(feature_sets_ints):

    features[i, -len(row):] = np.array(row)[:seq_len]

seq_len = 24

features_test = np.zeros((len(feature_sets_ints_test), seq_len), dtype=int)

for i, row in enumerate(feature_sets_ints_test):

    features_test[i, -len(row):] = np.array(row)[:seq_len]
split_frac = 0.75

split_idx = int(len(features)*split_frac)

train_x, val_x = features[:split_idx], features[split_idx:]

train_y, val_y = labels[:split_idx], labels[split_idx:]



test_idx = int(len(val_x)*0.5)

val_x, test_x = val_x[:test_idx], val_x[test_idx:]

val_y, test_y = val_y[:test_idx], val_y[test_idx:]



print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(train_x.shape), 

      "\nValidation set: \t{}".format(val_x.shape),

      "\nTest set: \t\t{}".format(test_x.shape))
lstm_size = 256

lstm_layers = 1

batch_len = 100

learning_rate = 0.001
n_words = len(vocab_to_int)+1



# Create the graph object

graph = tf.Graph()

# Add nodes to the graph

with graph.as_default():

    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')

    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    batch_size = tf.placeholder_with_default(tf.constant(batch_len), shape=[], name='batch_size')
n_words
# Size of the embedding vectors (number of units in the embedding layer)

embed_size = 300 



with graph.as_default():

    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))

    embed = tf.nn.embedding_lookup(embedding, inputs_)
with graph.as_default():

    # Your basic LSTM cell

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    

    # Add dropout to the cell

    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    

    # Stack up multiple LSTM layers, for deep learning

    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

    

    # Getting an initial state of all zerosn

    initial_state = cell.zero_state(batch_size, tf.float32)
with graph.as_default():

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,

                                             initial_state=initial_state)
with graph.as_default():

    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)

    cost = tf.losses.mean_squared_error(labels_, predictions)

    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
with graph.as_default():

    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
def get_batches(x, y, batch_size=100):

    n_batches = len(x)//batch_size

    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    for ii in range(0, len(x), batch_size):

        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
epochs = 5



with graph.as_default():

    saver = tf.train.Saver()



with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())

    iteration = 1

    for e in range(epochs):

        state = sess.run(initial_state)

        

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_len), 1):

            feed = {inputs_: x,

                    labels_: y[:, None],

                    keep_prob: 0.5,

                    initial_state: state}

            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            

            if iteration%5==0:

                print("Epoch: {}/{}".format(e, epochs),

                      "Iteration: {}".format(iteration),

                      "Train loss: {:.3f}".format(loss))



            if iteration%25==0:

                val_acc = []

                val_state = sess.run(cell.zero_state(batch_len, tf.float32))

                for x, y in get_batches(val_x, val_y, batch_len):

                    feed = {inputs_: x,

                            labels_: y[:, None],

                            keep_prob: 1,

                            initial_state: val_state}

                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)

                    val_acc.append(batch_acc)

                print("Val acc: {:.3f}".format(np.mean(val_acc)))

            iteration +=1

    saver.save(sess, "checkpoints/survival_preds.ckpt")
test_acc = []

with tf.Session(graph=graph) as sess:

    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    test_state = sess.run(cell.zero_state(batch_len, tf.float32))

    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_len), 1):

        feed = {inputs_: x,

                labels_: y[:, None],

                keep_prob: 1,

                initial_state: test_state}

        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)

        test_acc.append(batch_acc) 

    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
with tf.Session(graph=graph) as sess:

    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    test_state = sess.run(cell.zero_state(features_test.shape[0], tf.float32))

    feed = {inputs_: features_test,

            keep_prob: 1,

            initial_state: test_state}

    preds = np.asarray(sess.run([predictions], feed_dict=feed))

    preds =  np.where(preds >= 0.5, 1, preds)

    preds =  np.where(preds < 0.5, 0, preds)

    preds = np.asarray(preds, dtype=np.int32)

    #print(preds)

    df = pd.DataFrame(preds[0])

    df.index += 892 # test file passenger column start with this id

    df.columns = ['Survived']

    df.to_csv('titanic_test_kaggle_submission.csv', index_label='PassengerId')