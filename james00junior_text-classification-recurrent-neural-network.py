import numpy as np

import tensorflow as tf

import pandas as pd

from sklearn.preprocessing import LabelEncoder
import collections

import random

import time



class WordModel:

    

    def __init__(self, batch_size, dimension_size, learning_rate, vocabulary_size):

        

        self.train_inputs = tf.placeholder(tf.int32, shape = [batch_size])

        self.train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

        

        # randomly generated initial value for each word dimension, between -1.0 to 1.0

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, dimension_size], -1.0, 1.0))

        

        # find train_inputs from embeddings

        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        

        # estimation for not normalized dataset

        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, dimension_size], stddev = 1.0 / np.sqrt(dimension_size)))

        

        # each node have their own bias

        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        

        # calculate loss from nce, then calculate mean

        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_weights, biases = self.nce_biases, labels = self.train_labels,

                                                  inputs = embed, num_sampled = batch_size / 2, num_classes = vocabulary_size))

        

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        

        # normalize the data by simply reduce sum

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        

        # normalizing each embed

        self.normalized_embeddings = embeddings / self.norm



def read_data(filename):



    dataset = pd.read_csv(filename)

    rows = dataset.shape[0]

    print('there are', rows, 'total rows')

    

    # last column is our target

    label = dataset.ix[:, -1:].values

        

    # get second and third column values

    concated = []

    data = dataset.ix[:, 1:3].values

    

    for i in range(data.shape[0]):

        string = ""

            

        for k in range(data.shape[1]):

            string += data[i][k] + " "

            

        concated.append(string) 

    

    # get all split strings from second column and second last column

    dataset = dataset.ix[:, 1:3].values

    string = []

    for i in range(dataset.shape[0]):

        for k in range(dataset.shape[1]):

            string += dataset[i][k].split()

    

    return string, concated, label, list(set(string))



def build_dataset(words, vocabulary_size):

    count = []

    # extend count

    # sorted decending order of words

    count.extend(collections.Counter(words).most_common(vocabulary_size))



    dictionary = dict()

    for word, _ in count:

        #simply add dictionary of word, used frequently placed top

        dictionary[word] = len(dictionary)



    data = []

    unk_count = 0

    for word in words:

        if word in dictionary:

            index = dictionary[word]



        data.append(index)

    

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, dictionary, reverse_dictionary



def generate_batch_skipgram(words, batch_size, num_skips, skip_window):

    data_index = 0

    

    #check batch_size able to convert into number of skip in skip-grams method

    assert batch_size % num_skips == 0

    

    assert num_skips <= 2 * skip_window

    

    # create batch for model input

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)

    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1

    

    # a buffer to placed skip-grams sentence

    buffer = collections.deque(maxlen=span)

    

    for i in range(span):

        buffer.append(words[data_index])

        data_index = (data_index + 1) % len(words)

    

    for i in range(batch_size // num_skips):

        target = skip_window

        targets_to_avoid = [skip_window]

        

        for j in range(num_skips):

            

            while target in targets_to_avoid:

                # random a word from the sentence

                # if random word still a word already chosen, simply keep looping

                target = random.randint(0, span - 1)

            

            targets_to_avoid.append(target)

            batch[i * num_skips + j] = buffer[skip_window]

            labels[i * num_skips + j, 0] = buffer[target]

        

        buffer.append(words[data_index])

        data_index = (data_index + 1) % len(words)

    

    data_index = (data_index + len(words) - span) % len(words)

    return batch, labels



def generatevector(dimension, batch_size, skip_size, skip_window, num_skips, iteration, words_real):

    

    print('data size: ', len(words_real))

    data, dictionary, reverse_dictionary = build_dataset(words_real, len(words_real))

    

    sess = tf.InteractiveSession()

    print('Creating Word2Vec model.')

    

    model = WordModel(batch_size, dimension, 0.01, len(dictionary))

    sess.run(tf.global_variables_initializer())

    

    last_time = time.time()

    

    for step in range(iteration):

        new_time = time.time()

        batch_inputs, batch_labels = generate_batch_skipgram(data, batch_size, num_skips, skip_window)

        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

        

        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)

        

        if ((step + 1) % 1000) == 0:

            print('epoch: ', step + 1, ', loss: ', loss, ', speed: ', time.time() - new_time)

    

    tf.reset_default_graph()       

    return dictionary, reverse_dictionary, model.normalized_embeddings.eval()
dimension = 32

skip_size = 8

skip_window = 1

num_skips = 2

iteration_train_vectors = 5000



num_layers = 2

size_layer = 256

learning_rate = 0.001

epoch = 10

batch = 30
string, data, label, vocab = read_data('../input/train.csv')

label_encode = LabelEncoder().fit_transform(label)

dictionary, reverse_dictionary, vectors = generatevector(dimension, dimension, skip_size, skip_window, num_skips, iteration_train_vectors, string)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

sns.set()



embed = TSNE(n_components = 2).fit_transform(vectors)

plt.figure(figsize = (32, 32))



for i, label in enumerate(reverse_dictionary):

    x, y = embed[i, :]

    plt.scatter(x, y, c = 'g')

    plt.annotate(label, xy = (x, y), xytext = (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom')



plt.show()
bagofword = np.zeros((len(data), len(vocab)))

for i in range(len(data)):

    for _, text in enumerate(data[i].split()):

        bagofword[i, vocab.index(text)] += 1.0

        

from sklearn.decomposition import PCA

from sklearn.preprocessing import Normalizer



bagofword = Normalizer().fit_transform(bagofword)

data_visual = PCA(n_components = 2).fit_transform(bagofword)

palette = ['r', 'b']

data_label = ['negative', 'positive']



plt.figure(figsize = (18, 18))

for no, _ in enumerate(np.unique(label_encode)):

    plt.scatter(data_visual[label_encode == no, 0], data_visual[label_encode == no, 1], c = palette[no], label = data_label[no], alpha = 0.5)

    

plt.legend()

plt.show()
dimension_input = len(vocab)

print('dimension size: ', str(dimension_input))

print('sentence size: ', len(data))
class Model:

    

    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):

        

        def lstm_cell():

            return tf.nn.rnn_cell.LSTMCell(size_layer, activation = tf.nn.relu)

        

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        

        # [dimension of word, batch size, dimension input]

        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])

        

        #[batch size, dimension input]

        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        

        self.outputs, self.last_state = tf.nn.dynamic_rnn(self.rnn_cells, self.X, dtype = tf.float32)

        

        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))

        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))

        

        self.logits = tf.matmul(self.outputs[-1], self.rnn_W) + self.rnn_B

        

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))

        

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, label_encode, test_size = 0.15)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = Model(num_layers, size_layer, dimension_input, np.unique(label_encode).shape[0], learning_rate)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())
from sklearn import metrics



ACC_TRAIN, ACC_TEST, LOST = [], [], []

for i in range(epoch):

    total_cost = 0

    total_accuracy = 0

    last_time = time.time()

    

    for n in range(0, (len(X_train) // batch) * batch, batch):

        batch_x = np.zeros((dimension, batch, dimension_input))

        batch_y = np.zeros((batch, np.unique(Y_train).shape[0]))

        for k in range(batch):

            emb_data = np.zeros((dimension, dimension_input), dtype = np.float32)

            for _, text in enumerate(X_train[n + k].split()):

                # if the word got in the vocab

                try:

                    emb_data[:, vocab.index(text)] += vectors[dictionary[text], :]

                # if not, skip

                except:

                    continue



            batch_y[k, int(Y_train[n + k])] = 1.0

            batch_x[:, k, :] = emb_data[:, :]

            

        loss, _ = sess.run([model.cost, model.optimizer], feed_dict = {model.X : batch_x, model.Y : batch_y})

        total_accuracy += sess.run(model.accuracy, feed_dict = {model.X : batch_x, model.Y : batch_y})

        total_cost += loss

        

    total_cost /= (len(X_train) // batch)

    total_accuracy /= (len(X_train) // batch)

    times = (time.time() - last_time) / (len(X_train) // batch)

        

    ACC_TRAIN.append(total_accuracy)

    LOST.append(total_cost)

        

    print('epoch: ', i + 1, ', loss: ', total_cost, ', accuracy train: ', total_accuracy, 's / batch: ', times)

        

    batch_x = np.zeros((dimension, Y_test.shape[0], dimension_input))

    batch_y = np.zeros((Y_test.shape[0], np.unique(Y_test).shape[0]))

        

    for k in range(Y_test.shape[0]):

        emb_data = np.zeros((dimension, dimension_input), dtype = np.float32)

        for _, text in enumerate(X_test[k].split()):

            # if the word got in the vocab

            try:

                emb_data[:, vocab.index(text)] += vectors[dictionary[text], :]

            # if not, skip

            except:

                continue

                

        batch_y[k, int(Y_test[k])] = 1.0 

        batch_x[:, k, :] = emb_data[:, :]

            

    testing_acc, logits = sess.run([model.accuracy, tf.cast(tf.argmax(model.logits, 1), tf.int32)], feed_dict = {model.X : batch_x, model.Y : batch_y})

    print ('testing accuracy: ', testing_acc)

    ACC_TEST.append(testing_acc)

    print (metrics.classification_report(Y_test, logits, target_names = ['negative', 'positive']))

            

plt.subplot(1, 2, 1)

x_component = [i for i in range(len(LOST))]

plt.plot(x_component, LOST)

plt.xlabel('epoch')

plt.ylabel('loss')

plt.subplot(1, 2, 2)

plt.plot(x_component, ACC_TRAIN, label = 'train accuracy')

plt.plot(x_component, ACC_TEST, label = 'test accuracy')

plt.legend()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()