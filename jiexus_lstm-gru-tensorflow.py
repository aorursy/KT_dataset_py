# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)

df.head()

symbols = list(set(df.symbol))

len(symbols)
symbols[:20]
df['symbol'].value_counts()
df_fun = pd.read_csv("../input/fundamentals.csv")

#print(df_fun.columns)

df_fun.head()
df_fun.describe()
def get_stats(group):

    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}

bins = [0, 20, 50, 100, 1200]

group_names = ['Low', 'Okay', 'Good', 'Great']

df_fun['categories'] = pd.cut(df_fun['Cash Ratio'], bins, labels=group_names)



df_fun['Cash Ratio'].groupby(df_fun['categories']).apply(get_stats).unstack()
df_sec = pd.read_csv("../input/securities.csv")

print(df_sec.columns)

df_s = df_sec['Ticker symbol'].groupby(df_sec['GICS Sector'])

df_s.describe()

#sector = df_sec[df_sec['Ticker symbol'] == 'FB']['GICS Sector']

#print(sector)

#df_same = df_sec[df_sec['GICS Sector'] == sector ]

#def df_same_sec_symbols(df_sec, sym):

#    sector = df_sec[df_sec['Ticker symbol'] == sym]['GICS Sector']

#    return df_sec

#sector.head()

#s = sector.at[0,'GICS Sector']
df_gg = df[df.symbol == 'GOOG'].copy()

df_gg.drop(['symbol'],1,inplace=True)

df_gg.count()

#df_fb.sort_index(axis=1, inplace=True)

#df_fb.head()


plt.plot(df_gg.close)

plt.show()
n_steps = 28

n_inputs = 28

n_neurons = 150

n_outputs = 10

n_layers = 3



learning_rate = 0.001



X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

y = tf.placeholder(tf.int32, [None])



lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

              for layer in range(n_layers)]



#lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)

#gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)



multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

top_layer_h_state = states[-1][1]

logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    

init = tf.global_variables_initializer()
n_epochs = 10

batch_size = 150



with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        for iteration in range(mnist.train.num_examples // batch_size):

            X_batch, y_batch = mnist.train.next_batch(batch_size)

            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

        print("Epoch", epoch, "Train accuracy =", acc_train, "Test accuracy =", acc_test)