import tensorflow as tf

import pandas as pd 

import numpy as np

from string import punctuation

from collections import Counter

from tqdm import tqdm

%matplotlib inline
sentiment_data = pd.read_csv('../input/train.csv')

sentiment_data.head()
from sklearn.utils import shuffle

sentiment_data = shuffle(sentiment_data)
labels = sentiment_data.iloc[:, 0].values

reviews = sentiment_data.iloc[:, 1].values
reviews_processed = []

unlabeled_processed = [] 

for review in reviews:

    review_cool_one = ''.join([char for char in review if char not in punctuation])

    reviews_processed.append(review_cool_one)
word_reviews = []

all_words = []

for review in reviews_processed:

    word_reviews.append(review.lower().split())

    for word in review.split():

        all_words.append(word.lower())

    

counter = Counter(all_words)

vocab = sorted(counter, key=counter.get, reverse=True)

vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}
reviews_to_ints = []

for review in word_reviews:

    reviews_to_ints.append([vocab_to_int[word] for word in review])
reviews_lens = Counter([len(x) for x in reviews_to_ints])

print('Zero-length {}'.format(reviews_lens[0]))

print("Max review length {}".format(max(reviews_lens)))
seq_len = 250



features = np.zeros((len(reviews_to_ints), seq_len), dtype=int)

for i, review in enumerate(reviews_to_ints):

    features[i, -len(review):] = np.array(review)[:seq_len]
X_train = features[:6400]

y_train = labels[:6400]



X_test = features[6400:]

y_test = labels[6400:]



print('X_trian shape {}'.format(X_train.shape))
def weights_init(shape):

    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
def bias_init(shape):

    return tf.Variable(tf.zeros(shape=shape))
def define_inputs(batch_size, sequence_len):

    '''

    This function is used to define all placeholders used in the network.

    

    Input(s): batch_size - number of samples that we are feeding to the network per step

              sequence_len - number of timesteps in the RNN loop

              

    Output(s): inputs - the placeholder for reviews

               targets - the placeholder for classes (sentiments)

               keep_probs - the placeholder used to enter value for dropout in the model    

    '''

    inputs = tf.placeholder(tf.int32, [batch_size, sequence_len], name='inputs_reviews')

    targets = tf.placeholder(tf.float32, [batch_size, 1], name='target_sentiment')

    keep_probs = tf.placeholder(tf.float32, name='keep_probs')

    

    return inputs, targets, keep_probs
def embeding_layer(vocab_size, embeding_size, inputs):

    '''

    Function used for creating word embedings (word vectors)

    

    Input(s): vocab_size - number of words in the vocab

              embeding_size - length of a vector used to represent a single word from vocab

              inputs - inputs placeholder

    

    Output(s): embed_expended -  word embedings expended to be 4D tensor so we can perform Convolution operation on it

    '''

    word_embedings = tf.Variable(tf.random_uniform([vocab_size, embeding_size]))

    embed = tf.nn.embedding_lookup(word_embedings, inputs)

    embed_expended = tf.expand_dims(embed, -1) #expend dims to 4d for conv layer

    return embed_expended
def text_conv(input, filter_size, number_of_channels, number_of_filters, strides=(1, 1), activation=tf.nn.relu, max_pool=True):

    '''

    This is classical CNN layer used to convolve over embedings tensor and gether useful information from it.

    

    Input(s): input - word_embedings

              filter_size - size of width and height of the Conv kernel

              number_of_channels - in this case it is always 1

              number_of_filters - how many representation of the input review are we going to output from this layer 

              strides - how many pixels does kernel move to the side and up/down

              activation - a activation function

              max_pool - boolean value which will trigger a max_pool operation on the output tensor

    

    Output(s): text_conv layer

    

    '''

    weights = weights_init([filter_size, filter_size, number_of_channels, number_of_filters])

    bias = bias_init([number_of_filters])

    

    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding='SAME')

    

    if activation != None:

        layer = activation(layer)

    

    if max_pool:

        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2 ,1], strides=[1, 2, 2, 1], padding='SAME')

    

    return layer
def lstm_layer(lstm_size, number_of_layers, batch_size, dropout_rate):

    '''

    This method is used to create LSTM layer/s for PixelRNN

    

    Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer

              number_of_layers - used to define how many of LSTM layers do we want in the network

              batch_size - in this method this information is used to build starting state for the network

              dropout_rate - used to define how many cells in a layer do we want to 'turn off'

              

    Output(s): cell - lstm layer

               init_state - zero vectors used as a starting state for the network

    '''

    def cell(size, dropout_rate=None):

        layer = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        

        return tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)

            

    cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size, dropout_rate) for _ in range(number_of_layers)])

    

    init_state = cell.zero_state(batch_size, tf.float32)

    return cell, init_state
def flatten(layer, batch_size, seq_len):

    '''

    Used to transform/reshape 4d conv output to 2d matrix

    

    Input(s): Layer - text_cnn layer

              batch_size - how many samples do we feed at once

              seq_len - number of time steps

              

    Output(s): reshaped_layer - the layer with new shape

               number_of_elements - this param is used as a in_size for next layer

    '''

    dims = layer.get_shape()

    number_of_elements = dims[2:].num_elements()

    

    reshaped_layer = tf.reshape(layer, [batch_size, int(seq_len/2), number_of_elements])

    return reshaped_layer, number_of_elements
def dense_layer(input, in_size, out_size, dropout=False, activation=tf.nn.relu):

    '''

    Output layer for the lstm netowrk

    

    Input(s): lstm_outputs - outputs from the RNN part of the network

              input_size - in this case it is RNN size (number of neuros in RNN layer)

              output_size - number of neuros for the output layer == number of classes

              

    Output(s) - logits, 

    '''

    weights = weights_init([in_size, out_size])

    bias = bias_init([out_size])

    

    layer = tf.matmul(input, weights) + bias

    

    if activation != None:

        layer = activation(layer)

    

    if dropout:

        layer = tf.nn.dropout(layer, 0.5)

        

    return layer
def loss_optimizer(logits, targets, learning_rate, ):

    '''

    Function used to calculate loss and minimize it

    

    Input(s): rnn_out - logits from the fully_connected layer

              targets - targets used to train network

              learning_rate/step_size

    

    

    Output(s): optimizer - optimizer of choice

               loss - calculated loss function

    '''

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))

    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return loss, optimizer
class SentimentCNN(object):

    

    def __init__(self, learning_rate=0.001, batch_size=100, seq_len=250, vocab_size=10000, embed_size=300,

                conv_filters=32, conv_filter_size=5, number_of_lstm_layers=1, lstm_units=128):

        

        

        '''

        To created Sentiment embed network CNN-LSTM create object of this class.

        

        Input(s): learning_rate/step_size - how fast are we going to find global minima

                  batch_size -  the nuber of samples to feed at once

                  seq_len - the number of timesteps in unrolled RNN

                  vocab_size - the number of nunique words in the vocab

                  embed_size - length of word embed vectors

                  conv_filters - number of filters in output tensor from CNN layer

                  conv_filter_size - height and width of conv kernel

                  number_of_lstm_layers - the number of layers used in the LSTM part of the network

                  lstm_units - the number of neurons/cells in a LSTM layer

        

        '''

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_probs = define_inputs(batch_size, seq_len)

        

        embed = embeding_layer(vocab_size, embed_size, self.inputs)

        

        #Building the network

        convolutional_part = text_conv(embed, conv_filter_size, 1, conv_filters)

        conv_flatten, num_elements = flatten(convolutional_part, batch_size, seq_len)

        

        cell, init_state = lstm_layer(lstm_units, number_of_lstm_layers, batch_size, self.keep_probs)

        

        outputs, states = tf.nn.dynamic_rnn(cell, conv_flatten, initial_state=init_state)

        

        review_outputs = outputs[:, -1, :]

        

        logits = dense_layer(review_outputs, lstm_units, 1, activation=None)

        

        self.loss, self.opt = loss_optimizer(logits, self.targets, learning_rate)

        

        preds = tf.nn.sigmoid(logits)

        currect_pred = tf.equal(tf.cast(tf.round(preds), tf.int32), tf.cast(self.targets, tf.int32))

        self.accuracy = tf.reduce_mean(tf.cast(currect_pred, tf.float32))
model = SentimentCNN(learning_rate=0.001, 

                     batch_size=50, 

                     seq_len=250, 

                     vocab_size=len(vocab_to_int) + 1, 

                     embed_size=300,

                     conv_filters=32, 

                     conv_filter_size=5, 

                     number_of_lstm_layers=1, 

                     lstm_units=128)
session = tf.Session()
session.run(tf.global_variables_initializer())
epochs = 5

batch_size = 50

drop_rate = 0.7
for i in range(epochs):

    epoch_loss = []

    train_accuracy = []

    for ii in tqdm(range(0, len(X_train), batch_size)):

        X_batch = X_train[ii:ii+batch_size]

        y_batch = y_train[ii:ii+batch_size].reshape(-1, 1)

        

        c, _, a = session.run([model.loss, model.opt, model.accuracy], feed_dict={model.inputs:X_batch, 

                                                                                  model.targets:y_batch,

                                                                                  model.keep_probs:drop_rate})

        

        epoch_loss.append(c)

        train_accuracy.append(a)

        

    

    print("Epoch: {}/{}".format(i, epochs), " | Epoch loss: {}".format(np.mean(epoch_loss)), 

          " | Mean train accuracy: {}".format(np.mean(train_accuracy)))
test_accuracy = []



ii = 0

while ii + batch_size <= len(X_test):

    X_batch = X_test[ii:ii+batch_size]

    y_batch = y_test[ii:ii+batch_size].reshape(-1, 1)



    a = session.run([model.accuracy], feed_dict={model.inputs:X_batch, 

                                                 model.targets:y_batch, 

                                                 model.keep_probs:1.0})

    

    test_accuracy.append(a)

    ii += batch_size
print("Test accuracy: {}".format(np.mean(test_accuracy)))
session.close()