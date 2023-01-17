# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

np.set_printoptions(threshold=np.inf)

import operator

import nltk

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df.head()
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df = df.rename(columns={'v1': 'class', 'v2':'text'})

df.head()
from nltk.tokenize import WhitespaceTokenizer

tokeniser = WhitespaceTokenizer()





def tokenize(sentence):

    return tokeniser.tokenize(sentence)

num_top_words = 1000



all_words = {}



def build_words(string_in):

    for w in tokenize(string_in):

        all_words[w] = all_words.get(w, 0) + 1



for x in df['text']:

    build_words(x)



sorted_words = sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)

sorted_words = list(map(lambda x: x[0], sorted_words))

sorted_words = sorted_words[:num_top_words]



words_by_emails = []



def count_words_per_email(text):

    row = np.zeros(len(sorted_words))  # Add the label column

    for word in tokenize(text):

        try:

            row[sorted_words.index(word)] = row[sorted_words.index(word)] + 1

        except ValueError:

            pass

    return row



X_rows = []

for _row in df['text']:

    X_rows.append(count_words_per_email(_row))

X_rows = np.array(X_rows)



print(X_rows.shape)    
_labels = df['class'].map(lambda x: 1.0 if x == 'spam' else 0.0).values

y_labels = tf.one_hot(_labels, depth=2)
learning_rate = 0.003

n_h1 = 512

n_h2 = 256



n_class = 2

batch_size = 64

epochs = 5

X = tf.placeholder("float", [None, X_rows.shape[1]])

Y = tf.placeholder('float', [None, n_class])



weights = {

    'w0': tf.get_variable("w0", shape=[X_rows.shape[1], n_h1], initializer=tf.contrib.layers.xavier_initializer()),

    'w1': tf.get_variable("w1", shape=[n_h1, n_h2], initializer=tf.contrib.layers.xavier_initializer()),

    'w_out': tf.get_variable("w_out", shape=[n_h2, n_class], initializer=tf.contrib.layers.xavier_initializer())

}



bias = {

    'b0': tf.Variable(tf.random_normal([n_h1])),

    'b1': tf.Variable(tf.random_normal([n_h2])),

    'b_out': tf.Variable(tf.random_normal([n_class])),

}



keep_prob_1 = tf.placeholder(tf.float32)

keep_prob_2 = tf.placeholder(tf.float32)
def neural_net(x):

    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['w0']), bias['b0']))

    layer_1_drop = tf.nn.dropout(layer_1, keep_prob_1)

    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1_drop, weights['w1']), bias['b1']))

    layer_2_drop = tf.nn.dropout(layer_2, keep_prob_2)

    out = tf.add(tf.matmul(layer_2_drop, weights['w_out']), bias['b_out'])

    return out

logits = neural_net(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    labels = sess.run(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(X_rows, labels, test_size=0.1)

    num_batches = int(len(X_train) / batch_size)

    for _e in range(epochs):

        for step in range(num_batches):

            feed_dict={

                 X: X_train[step * batch_size: (step + 1) * batch_size],

                 Y: y_train[step * batch_size: (step + 1) * batch_size],

                 keep_prob_1: 0.8,

                 keep_prob_2: 0.8

            }

            sess.run(train_op, feed_dict=feed_dict)

            if step == 0 or step % 10 == 0:

                loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict)

                print('Step: {0}, Loss={1}, Training Accuracy={2}'.format(

                    step, loss, acc

                ))

    # Test

    num_batches = int(len(X_test) / batch_size)

    total_acc = 0

    for step in range(num_batches):

        feed_dict={

             X: X_test[step * batch_size: (step + 1) * batch_size],

             Y: y_test[step * batch_size: (step + 1) * batch_size],

             keep_prob_1: 1.0,

             keep_prob_2: 1.0

        }

        acc = sess.run(accuracy, feed_dict=feed_dict)

        total_acc = total_acc + acc

        print('Average Test Accuracy = {0}'.format(total_acc/(step + 1)))

        



        