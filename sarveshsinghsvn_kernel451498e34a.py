# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import tensorflow as tf

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/fasttext-crawl-300d-2m"))
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'



x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)

    

def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            #print("NOT found")

            pass

    return embedding_matrix
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train) + list(x_test))

MAX_LEN = 220

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)



x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)



embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
batch_size = 100

lstm_size = 256

num_input = 600

num_hidden = 1024



seq_max_len = 220

input_dim = 600



out_dim = 1             # output dimension



# Parameters

learning_rate = 0.01    # The optimization initial learning rate

training_steps = 1  # Total number of training steps

batch_size = 10         # batch size

display_freq = 1     # Frequency of displaying the training results

#num_hidden_units = 10   # number of hidden units
# weight and bais wrappers

def weight_variable(shape):

    initer = tf.truncated_normal_initializer(stddev=0.01)

    return tf.get_variable('W',dtype=tf.float32,shape=shape,initializer=initer)



def bias_variable(shape):

    initial = tf.constant(0., shape=shape, dtype=tf.float32)

    return tf.get_variable('b',dtype=tf.float32,initializer=initial)
def RNN(x, weights, biases, n_hidden, seq_max_len, seq_len):



    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)



    # Hack to build the indexing and retrieve the right output.

    batch_size = tf.shape(outputs)[0]

    # Start indices for each sample

    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)

    # Indexing

    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    out = tf.matmul(outputs, weights) + biases

    return out
# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)

x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])

seqLen = tf.placeholder(tf.int32, [None])

y = tf.placeholder(tf.float32, [None, 1])
# create weight matrix initialized randomly from N~(0, 0.01)

W = weight_variable(shape=[num_hidden, out_dim])



# create bias vector initialized as zero

b = bias_variable(shape=[out_dim])



# Network predictions

pred_out = RNN(x, W, b, num_hidden, seq_max_len, seqLen)
# Define the loss function (i.e. mean-squared error loss) and optimizer

cost = tf.reduce_mean(tf.square(pred_out - y))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Creating the op for initializing all variables

init = tf.global_variables_initializer()



y_train = y_train.reshape((y_train.size,1))
with tf.Session() as sess:

    sess.run(init)

    print('----------Training---------')

    seq_len_batch = np.full((100, ), 220)

    for i in range(training_steps):

        sum_mse = 0

        if(i==1):

            tem = 18048

        else:

            tem = 9000

        for j in range(tem):

        #for j in range(2):

            x_batch = embedding_matrix[x_train[j*100:j*100+100],:]

            y_batch = y_train[j*100:j*100+100]         

            _, mse = sess.run([train_op, cost], feed_dict={x: x_batch,

                                                           y: y_batch,

                                                           seqLen: seq_len_batch})

            sum_mse = mse + sum_mse

        if i % display_freq == 0:

            print('Step {0:<6}, MSE={1:.4f}'.format(i, sum_mse))

            



    y_test = np.full((97320,),0.0)

    for k in range(973):

        x_batch = embedding_matrix[x_test[k*100:k*100+100],:]

        temp1 = sess.run([pred_out], feed_dict={x: x_batch,seqLen: seq_len_batch})

        #print(temp.shape)

        #print(temp1[0][1])

        y_test[k*100:k*100+100] = temp1[0][:].ravel()

        #print(y_test[k*100:k*100+100])

    submission = pd.DataFrame.from_dict({'id': test_df.id,'prediction': y_test})    

    submission.to_csv('submission.csv', index=False)
#submission.to_csv('submission.csv', index=False)