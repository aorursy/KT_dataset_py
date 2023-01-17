# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import random

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/all.csv')
content = df['content'].tolist()[:3]
def sent_to_words(content):
    return [np.array([x.split() for x in poem.split()]) for poem in content]
poems = sent_to_words(content)
def build_dict(poems):
    dictionary = {}
    rev_dict = {}
    count = 0
    for content in poems:
        for i in content:
            if i[0] in dictionary:
                pass
            else:
                dictionary[i[0]] = count
                count += 1
    rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, rev_dict
dictionary, rev_dict = build_dict(poems)
import tensorflow as tf
from tensorflow.contrib import rnn
vocab_size = len(dictionary)
# Parameters
learning_rate = 0.0001
training_iters = 1600
display_step = 200
n_input = 9
# number of units in RNN cell
n_hidden = 512
# tf Graph input
tf.device("/device:GPU:0")
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}
def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
pred = RNN(x, weights, biases)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
init = tf.global_variables_initializer()
df_train = sent_to_words(content)
j = 0
for i in df_train:
    if i.shape[0] <= n_input:
        df_train = np.delete(df_train, (j), axis = 0)
        j -= 1
    j += 1
with tf.Session() as session:
    session.run(init)
    step = 0
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    while step < training_iters:
        acc_total = 0
        loss_total = 0
        j = 0
        for training_data in df_train:
            m = training_data.shape[0]
            windows = m - n_input
            acc_win = 0
            for window in range(windows):
                batch_x = training_data[window : window + n_input]
                batch_y = training_data[window + n_input]
                symbols_in_keys = [dictionary[i[0]] for i in batch_x]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        
                symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                symbols_out_onehot[dictionary[batch_y[0]]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                loss_total += loss
                acc_win += acc
            acc_total += float(acc_win) / m
        acc_total /= len(df_train)
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total))
        step += 1
    print("Optimization Finished!")
    save_path = saver.save(session, "../working/model.ckpt")
    print("Model saved in path: %s" % save_path)
    while True:
        prompt = "%s words: " % n_input
        sentence = 'When I Queen Mab within my fancy viewed, My'
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(64):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,rev_dict[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
            break
        except:
            print("Word not in dictionary")


