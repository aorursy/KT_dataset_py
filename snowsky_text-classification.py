from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

import time

import collections

import numpy as np

import random

import tensorflow as tf
class VecModel:

    

    def __init__(self, batch_size, dimension_size, learning_rate, vocabulary_size):

        

        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])

        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        

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

                                                  inputs=embed, num_sampled = batch_size / 2, num_classes = vocabulary_size))

        

        #for a small neural network, for me, Adam works the best

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        

        # normalize the data by simply reduce sum

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        

        # normalizing each embed

        self.normalized_embeddings = embeddings / self.norm
def read_data(filename, train = True):

    

    import pandas as pd

    dataset = pd.read_csv(filename)

    

    # 0 shape to get total of rows, 1 to get total of columns

    rows = dataset.shape[0]

    print ("there are ", rows, " total rows\n")

    

    if train:

        # get value on last column only

        label = dataset.ix[:, -1:].values

        

        # get all value except last column

        concated = []

        data = dataset.ix[:, 1:-1].values

        

        for i in range(data.shape[0]):

            string = ""

            

            for k in range(data.shape[1]):

                string += data[i][k] + " "

            

            concated.append(string)

                

    

    # get second column and second last column

    dataset = dataset.ix[:, 1:-1].values

    string = []

    for i in range(dataset.shape[0]):

        for k in range(dataset.shape[1]):

            string += dataset[i][k].split()

            

    string = list(set(string))

    

    if train:

        return string, concated, label

    else:

        return string
def build_dataset(words, vocabulary_size):

    count = []

    # extend count

    # -1 because first space used to place UNK keyword

    # sorted decending order of words

    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))



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

    return data, count, reverse_dictionary
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

    

    print ("Data size: ", len(words_real))



    data, _, dictionary = build_dataset(words_real, len(words_real))

    

    sess = tf.InteractiveSession()

    print ("Creating Word2Vec model..")

    

    model = VecModel(batch_size, dimension, 0.1, len(dictionary))

    sess.run(tf.global_variables_initializer())

    

    last_time = time.time()

    

    for step in range(iteration):

        batch_inputs, batch_labels = generate_batch_skipgram(data, batch_size, num_skips, skip_window)

        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

        

        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)

        

        if ((step + 1) % 100) == 0:

            

            new_time = time.time()

            diff = new_time - last_time

            last_time = new_time

            print ("batch: ", step + 1, ", loss: ", loss, ", speed: ", 100.0 / diff, " batches / s")

            

    return dictionary, model.normalized_embeddings.eval()
class Model:

    

    def __init__(self, activation, num_layers, size_layer, dimension, biased_node, learning_rate):

        

        if activation == 'relu':

            self.activation = tf.nn.relu

        elif activation == 'sigmoid':

            self.activation = tf.nn.sigmoid

        elif activation == 'tanh':

            self.activation = tf.nn.tanh

        else:

            raise Exception("model type not supported")

        

        self.X = tf.placeholder("float", [None, dimension])

        self.Y = tf.placeholder("float", [None, 1])

        

        self.size_relay = tf.placeholder(tf.int32, None)

        

        self.input_layer = tf.Variable(tf.random_normal([dimension, size_layer]))

    

        if biased_node:

            self.biased_input_layer = tf.Variable(tf.random_normal([size_layer]))

            self.biased = []

            for i in range(num_layers):

                self.biased.append(tf.Variable(tf.random_normal([size_layer])))

            

        self.layers = []

        for i in range(num_layers):

            self.layers.append(tf.Variable(tf.random_normal([size_layer, size_layer])))



        self.output_layer = tf.Variable(tf.random_normal([size_layer, 1]))

        

            

        if biased_node:

            self.first_l = self.activation(tf.add(tf.matmul(self.X[self.size_relay : self.size_relay + 1, :], self.input_layer), self.biased_input_layer))

            

            # prevent overfitting

            self.first_l = tf.nn.dropout(self.first_l, 1.0)

            

            self.next_l = self.activation(tf.add(tf.matmul(self.first_l, self.layers[0]), self.biased[0]))

                

            for i in range(1, num_layers - 1):

                self.next_l = self.activation(tf.add(tf.matmul(self.next_l, self.layers[i]), self.biased[i]))

                    

        else:

            self.first_l = self.activation(tf.matmul(self.X[i : i + 1, :], self.input_layer))

            

            # prevent overfitting

            self.first_l = tf.nn.dropout(self.first_l, 1.0)

            

            self.next_l = self.activation(tf.matmul(self.first_l, self.layers[0]))

                

            for i in range(1, num_layers - 1):

                self.next_l = self.activation(tf.matmul(self.next_l, self.layers[i]))

        

        self.first_cost = tf.reduce_mean(tf.pow(tf.transpose(self.layers[0]) - self.next_l, 2))

            

        self.first_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.first_cost)

         

        if biased_node:

            self.first_l = self.activation(tf.add(tf.matmul(self.X[-1 : , :], self.input_layer), self.biased_input_layer))

            

            # prevent overfitting

            self.first_l = tf.nn.dropout(self.first_l, 1.0)

            

            self.next_l = self.activation(tf.add(tf.matmul(self.first_l, self.layers[0]), self.biased[0]))

            

            for i in range(1, num_layers - 1):

                self.next_l = self.activation(tf.add(tf.matmul(self.next_l, self.layers[i]), self.biased[i]))

        

        else:

            self.first_l = self.activation(tf.matmul(self.X[-1 : , :], self.input_layer))

            

            # prevent overfitting

            self.first_l = tf.nn.dropout(self.first_l, 0.5)

            

            self.next_l = self.activation(tf.matmul(self.first_l, self.layers[0]))

            

            for i in range(1, num_layers - 1):

                self.next_l = self.activation(tf.matmul(self.next_l, self.layers[i]))

        

        # prevent overfitting

        self.next_l = tf.nn.dropout(self.next_l, 1.0)

        

        self.b = tf.Variable(tf.random_normal([1], mean = 0.0, stddev = 0.1))

        

        self.last_l = tf.add(tf.matmul(self.next_l, self.output_layer), self.b)

        

        self.second_cost = tf.reduce_mean(tf.square(self.last_l - self.Y))

        

        # calculate penalty for high value in nodes

        regularizers = tf.nn.l2_loss(self.input_layer) + sum(map(lambda x: tf.nn.l2_loss(x), self.layers)) + tf.nn.l2_loss(self.output_layer)

        

        # by simply cost the penalty value to prevent overfitting

        self.second_cost = tf.reduce_mean(self.second_cost + 0.1 * regularizers)

        

        self.second_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.second_cost)

        

        self.output = self.last_l

        

    

    def step(self, sess, X, Y):

    

        out, _, cost, _, _ = sess.run([self.output, self.second_optimizer, self.second_cost, self.first_optimizer, self.first_cost], feed_dict={self.size_relay: X.shape[0] - 1, self.X: X, self.Y: Y})

        

        return out, cost
dataset = '../input/train.csv'



# these global variables for vectors model

dimension = 32

skip_size = 8

skip_window = 1

num_skips = 2

iteration_train_vectors = 1000



# these global variables for NN

num_layers = 1

size_layer = 64

learning_rate = 0.05

epoch = 50

biased_node = True



split_data = 0.8



# got sigmoid, relu, tanh

activation = 'tanh'



# you can use word vectors or one-hot-encoder

use_word_vector = True
string, data, label = read_data(dataset, train = True)



label = LabelEncoder().fit_transform(label)



if use_word_vector:

    vocab, vectors = generatevector(dimension, dimension, skip_size, skip_window, num_skips, iteration_train_vectors, string)

else:

    vocab = list(set(string))

    dimension = len(vocab)

    print ("dimension size: ", dimension)



sess = tf.InteractiveSession()



data_train, data_test, label_train, label_test = train_test_split(data, label, test_size = split_data)



model = Model(activation, num_layers, size_layer, dimension, biased_node, learning_rate)



sess.run(tf.global_variables_initializer())
for n in range(epoch):

    total_cost = 0

    total_accuracy = 0

    last_time = 0.0

    

    for i in range(len(data_train)):

        last_time = time.time()

        data_ = data[i].split()

        

        if use_word_vector:

            emb_data = np.zeros((len(data_), vectors.shape[1]), dtype = np.float32)

            

            for x in range(len(data_)):

                found = False

                for id_, word in vocab.items():

                    

                    if word == data_[x]:

                        emb_data[x, :] = vectors[id_, :]

                        found = True

                        break

                

                if not found:

                    print ("not found any embedded words, recheck vectors pipelining, exiting..")

                    exit(0)

            

        else:

            emb_data = np.zeros((len(data_), len(vocab)), dtype = np.float32)

            

            for k, text in enumerate(data[i].split()):

                emb_data[k, vocab.index(text)] = 1.0

    

        X = np.array(emb_data)

        

        out, loss = model.step(sess, X, np.array([[label_train[i]]]))

        if int(round(out[0][0])) == label_train[i]:

            total_accuracy += 1

        total_cost += loss

        

    

    diff = time.time() - last_time

    print ("total accuracy during training: ", total_accuracy / (len(data_train) * 1.0))

    print ("batch: ", n + 1, ", loss: ", total_cost / len(data_train), ", speed: ", diff / len(data_train), " s / epoch")

    total_cost = 0

    total_accuracy = 0