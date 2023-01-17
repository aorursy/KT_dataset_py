import pandas as pd
import numpy as np
poetry = pd.read_csv('../input/all.csv')
num_poems = 7
poem = poetry['content'][:num_poems]
temp = ''
for i in range(num_poems):
    temp += poem[i] + '\r\n'
poem = temp
poem = poem.replace('\r\n\r\n', '\r\n')
poem = poem.replace('\r\n', ' ')
poem = poem.replace('\'', '')
import re
poem = re.sub(' +',' ',poem)
poem = poem.split()
words = list(set(poem))
vocab_size = len(words)
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
strdict = {}
revdict = {}
i = 0
for word in words:
    row = [0]
    col = [i]
    data = [1]
    strdict[word] =csr_matrix((data, (row, col)), shape=(1, vocab_size))
    revdict[i] = word
    i += 1
def convert_to_df(start, size, seq, vsize, batch_size):
    word_count = len(seq)
    inp = np.array([])
    out = np.array([])
    for i in range(batch_size):
        if start >= word_count - size:
            break
        ones = seq[start:start + size]
        inp_vector = vstack([strdict[x] for x in ones])
        out_vec = strdict[seq[start + size]]
        if i == 0:
            inp = inp_vector.toarray()
            out = out_vec.toarray()
        else:
            inp = np.dstack((inp, inp_vector.toarray()))
            out = np.vstack((out, out_vec.toarray()))
        start += 1
    inp = np.swapaxes(inp, 2, 0)
    inp = np.swapaxes(inp, 1, 2)
    return inp, out
len(poem)
import tensorflow as tf
def lstm_cell(weights, biases, prev_c, prev_h, curr_x):
    ft = tf.nn.sigmoid(tf.matmul(prev_h, weights['Wfh']) + tf.matmul(curr_x, weights['Wfx']) + biases['bf'])
    it = tf.nn.sigmoid(tf.matmul(prev_h, weights['Wih']) + tf.matmul(curr_x, weights['Wix']) + biases['bi'])
    Ct = tf.tanh(tf.matmul(prev_h, weights['Wch']) + tf.matmul(curr_x, weights['Wcx']) + biases['bc'])
    ot = tf.nn.sigmoid(tf.matmul(prev_h, weights['Woh']) + tf.matmul(curr_x, weights['Wox']) + biases['bo'])
    ct = tf.multiply(ft, prev_c) + tf.multiply(it, Ct)
    ht = tf.multiply(ot, tf.tanh(ct))
    return ct, ht
def lstm_layer(xin, weights, biases,cin, hin,num_inp):
    h = hin
    c  = cin
    next_inp = []
    for i in range(num_inp):
        curr = xin[:, i]
        curr = tf.cast(curr, tf.float32)
        h = tf.cast(h, tf.float32)
        c = tf.cast(c, tf.float32)
        c, h = lstm_cell(weights, biases, c, h, curr)
        next_inp.append(h)
    next_inp = tf.stack(next_inp)
    next_inp = tf.transpose(next_inp, [1, 0, 2])
    return next_inp
learning_rate = 0.001
training_iters = 1200
display_step =  300
num_inp = 4
n_hidden = 3
m = len(poem)
batch_size = 128
x = tf.placeholder(tf.float64, [None, num_inp, vocab_size])
y = tf.placeholder(tf.float64, [None, vocab_size])
weights = [None] * n_hidden
biases = [None] * n_hidden
for i in range(n_hidden):
    weights[i] = {
        'Wfh': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wfx': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wih': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wix': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wch': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wcx': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Woh': tf.Variable(tf.random_normal((vocab_size, vocab_size))),
        'Wox': tf.Variable(tf.random_normal((vocab_size, vocab_size)))
    }
    biases[i] = {
        'bf' : tf.Variable(tf.zeros([1, vocab_size])),
        'bi' : tf.Variable(tf.zeros([1, vocab_size])),
        'bc' : tf.Variable(tf.zeros([1, vocab_size])),
        'bo' : tf.Variable(tf.zeros([1, vocab_size]))
    }
cin = tf.zeros((1, vocab_size))
hin = tf.zeros((1, vocab_size))
tf.device("/device:GPU:0")
prev_inp = x
for i in range(1, n_hidden + 1):
    prev_inp = lstm_layer(prev_inp, weights[i - 1], biases[i - 1], cin, hin, num_inp)
dense_layer1 = prev_inp[:, -1]
pred = tf.nn.softmax(dense_layer1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer1, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    window_size = m - num_inp
    for epoch in range(training_iters):
        cst_total = 0
        acc_total = 0
        total_batches = int(np.ceil(window_size / batch_size))
        for i in range(total_batches):
            df_x, df_y = convert_to_df(i, num_inp, poem, vocab_size, batch_size)
            _, cst, acc = sess.run([optimizer, cost, accuracy], feed_dict = {x : df_x, y : df_y})
            cst_total += cst
            acc_total += acc
        if (epoch + 1) % display_step == 0:
            print('After ', (epoch + 1), 'iterations: Cost = ', cst_total / total_batches, 'and Accuracy: ', acc_total * 100 / total_batches, '%' )
    print('Optimiation finished!!!')
    #save_path = saver.save(sess, "../working/model.ckpt")
    #print("Model saved in path: %s" % save_path)
    print("Lets test")
    sentence = 'If my imagination from'
    sent = sentence.split()
    sent = vstack([strdict[s] for s in sent])
    sent  = sent.toarray()
    for q in range(64):
        one_hot = sess.run(pred, feed_dict={x : sent.reshape((1, num_inp, vocab_size))})
        index = int(tf.argmax(one_hot, 1).eval())
        sentence = "%s %s" % (sentence,revdict[index])
        sent = sent[1:]
        sent = np.vstack((sent, one_hot))
    print(sentence)

