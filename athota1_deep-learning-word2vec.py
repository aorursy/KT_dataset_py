import numpy as np 

import pandas as pd

from gensim.models import Word2Vec

import gensim

import os

import matplotlib.pyplot as plt

import tensorflow as tf
# Load Dataset

model = gensim.models.KeyedVectors.load_word2vec_format("../input/GoogleNews-vectors-negative300.bin", limit = 10000, binary=True)
stop_words = ["is", "the", "of", "and", "are", "like","has"]

corpus = ['apple is fruit',

          'banana is fruit',

          'peach is fruit',

          'the color of apple is red',

          'the skin of banana is yellow',

          'peach is pink',

          'fruit has color',

          'monkey like yellow banana',

          'kids like red apple',

          'monkey like pink peach',

          'yellow red and pink are color',

         ]
def remove_stop_words(corpus):

    corpus_with_out_stopwords = []

    for text in corpus:

        tmp = text.split(' ')

        for stop_word in stop_words:

            if stop_word in tmp:

                tmp.remove(stop_word)

        corpus_with_out_stopwords.append(" ".join(tmp))

        

    return corpus_with_out_stopwords
print('Actual corpus')

corpus
# Remove stopwords from corpus

corpus = remove_stop_words(corpus)
print('Formatted corpus: Corpus without stopwords')

corpus
# function to convert numbers to one hot vectors

def to_one_hot_encoding(data_point_index):

    one_hot_encoding = np.zeros(ONE_HOT_DIM)

    one_hot_encoding[data_point_index] = 1

    return one_hot_encoding
words = []

for text in corpus:

    for word in text.split(' '):

        words.append(word)

        

words = set(words)
# Data frame Generation

word2int = {}

data = []

sentences = []

WINDOW_SIZE = 2

for i,word in enumerate(words):

    word2int[word] = i

    

for sentence in corpus:

    sentences.append(sentence.split())

    

for sentence in sentences:

    for idx, word in enumerate(sentence):

        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence) + 1)]:

            if neighbor !=word:

                data.append([word, neighbor])

df = pd.DataFrame(data, columns = ['input', 'label'])
df.head(10)
df.shape
word2int
# word embedding will be 2 dimension for 2d visualization

EMBEDDING_DIM = 2

ONE_HOT_DIM = len(words)
X = [] # input word

Y = [] # target word



for x, y in zip(df['input'], df['label']):

    X.append(to_one_hot_encoding(word2int[ x ]))

    Y.append(to_one_hot_encoding(word2int[ y ]))



# convert X,Y to numpy arrays

X_train = np.asarray(X)

Y_train = np.asarray(Y)
# Define Tensorflow Graph

# Making placeholders for X_train and Y_train

x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# Hidden layer: which represents word vector eventually

W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))

b1 = tf.Variable(tf.random_normal([1])) #bias

hidden_layer = tf.add(tf.matmul(x,W1), b1)

# Output layer

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))

b2 = tf.Variable(tf.random_normal([1]))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# Loss function: cross entropy

loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# Training operation

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# Training

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init) 



iteration = 20000

for idx in range(iteration):

    # input is X_train which is one hot encoded word

    # label is Y_train which is one hot encoded neighbor word

    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})

    print('For loop Idx : ', idx, ' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))
vectors = sess.run(W1 + b1)
# Prepare table with word vector

word2vect_dataframe = pd.DataFrame(vectors, columns = ['a', 'b'])

word2vect_dataframe['word'] = words

word2vect_dataframe = word2vect_dataframe[['word', 'a', 'b']]
word2vect_dataframe
# Plot Word Vector in 2D Chart

fig, ax = plt.subplots()

for word, a, b in zip(word2vect_dataframe['word'], word2vect_dataframe['a'], word2vect_dataframe['b']):

    ax.annotate(word, (a ,b))    

PADDING = 1.0

x_axis_min = np.amin(vectors, axis=0)[0] - PADDING

y_axis_min = np.amin(vectors, axis=0)[1] - PADDING

x_axis_max = np.amax(vectors, axis=0)[0] + PADDING

y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

 

plt.xlim(x_axis_min,x_axis_max)

plt.ylim(y_axis_min,y_axis_max)

plt.rcParams["figure.figsize"] = (15,15)

plt.show()