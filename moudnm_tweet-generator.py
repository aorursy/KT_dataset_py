"""

necessary modules

"""



import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.utils import np_utils

import tensorflow.keras.utils as utils

from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding

import numpy as np



import random
"""

Use twitterscraper to scrape to json file

"""



import twitterscraper

from twitterscraper import query_tweets, query_tweets_from_user



import json

import re



def scrape_tweet(user, limit):

    scraped_tweets = query_tweets_from_user(user=user, limit=limit)



    data = {}

    data['text'] = []

    i = 0

    while (len(data['text']) < 1000 and i < len(scraped_tweets)):



        if scraped_tweets[i].is_retweet == 0:

            data['text'].append(scraped_tweets[i].text)

        i += 1



    with open('output.json', 'w') as outfile:

        json.dump(data, outfile)





"""

Load json file and combine all tweets

into one big text

"""





def load_tweets(input_file):

    # load data

    with open(input_file) as json_file:

        datafile = json.load(json_file)



    # remove links first

    for i in range(0, len(datafile['text'])):

        text = datafile['text'][i]

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        text = re.sub(r'pic.twitter.com\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        datafile['text'][i] = text



    # join all strings

    all_text = ' '.join(datafile['text'])

#     all_text = all_text[:int(len(all_text) / 10)] # change dataset size



    # sort characters

    vocab = sorted(set(all_text))

    return all_text, vocab





# scrape tweets

scrape_tweet('BillGates',1000)
# load data

all_text, vocab = load_tweets('output.json')

vocab_len = len(vocab)



print(all_text)

print(vocab)  
"""

Vectorize text

Create a mapping

"""



char2idx = dict((c, i) for i, c in enumerate(vocab))  # char to index

idx2char = dict((i, c) for i, c in enumerate(vocab))  # index to char



print(char2idx)

print(idx2char)
"""

Prepare input and output pairs

"""



# Prepare dataset of input to output pairs

def prepare_input_output(all_text, seq_length, char2idx):

    x_data = []  # training

    y_data = []  # labels



    for i in range(0, len(all_text) - seq_length, 1):

        # Define input and output sequences

        # Input is the current character plus desired sequence length

        in_seq = all_text[i:i + seq_length]



        # Out sequence is the initial character plus total sequence length

        out_seq = all_text[i + seq_length]



        # We now convert list of characters to integers based on

        # previously and add the values to our lists

        x_data.append([char2idx[char] for char in in_seq])

        y_data.append(char2idx[out_seq])



    return x_data, y_data



seq_length = 100

x_data = []  # training

y_data = []  # labels



x_data, y_data = prepare_input_output(all_text, seq_length, char2idx)

print(x_data)
"""

Convert to processed numpy array

"""

n_patterns = len(x_data)

X = np.reshape(x_data, (n_patterns, seq_length, 1))

X = X / float(vocab_len)



y = utils.to_categorical(y_data)  # one-hot encode
"""

build model

"""    

    

# model specs

batch_size = 64

epochs = 50

rnn_units = 256



def build_model(rnn_units, X, y):

    model = tf.keras.Sequential()

    model.add(LSTM(rnn_units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(rnn_units))

    # model.add(LSTM(rnn_units, return_sequences=True))

    model.add(Dropout(0.2))

    # model.add(LSTM(int(rnn_units / 2)))

    # model.add(Dropout(0.2))

    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')



    return model



model = build_model(rnn_units, X, y)

model.summary()
"""

Train model

"""



def train_model(model, X, y, epochs, batch_size):

    # Checkpoint and file saving

    filepath = "new_gates.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    desired_callbacks = [checkpoint]



#     # Avoid errors

#     from tensorflow.core.protobuf import rewriter_config_pb2



#     config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

#     off = rewriter_config_pb2.RewriterConfig.OFF

#     config_proto.graph_options.rewrite_options.arithmetic_optimization = off



#     sess = tf.Session(config=config_proto)



    # Train

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=desired_callbacks)

    

# train_model(model, X, y, 1, batch_size)
"""

Generate text

"""





def generate_text(model, weights_file, len_to_generate, x_data, vocab_len, idx2char):

    # load the network weights

    model.load_weights(weights_file)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    # pick a random seed

    start = np.random.randint(0, len(x_data) - 1)

    pattern = x_data[start]

    # print(pattern)

    # print("Seed:")

    # print("\"", ''.join([idx2char[value] for value in pattern]), "\"")



    # generate characters

    output = ''

    for i in range(len_to_generate):

        x = np.reshape(pattern, (1, len(pattern), 1))

        print(x.shape)

        x = x / float(vocab_len)

        prediction = model.predict(x, verbose=0)

        # index = np.argmax(prediction)

        index = np.random.choice(len(prediction[0]), p=prediction[0])



        # p = prediction.flatten()

        # index = np.random.choice(len(p), p=p)



        result = idx2char[index]

        # seq_in = [idx2char[value] for value in pattern]

        # print(result)

        output += result



        pattern.append(index)

        pattern = pattern[1:len(pattern)]



    print("Text Generated: {}".format(output))

    return output



import os

print(os.listdir("../input")) 

input_weights = ("../input/new_gates.hdf5")



generate_text(model, input_weights, 100, x_data, vocab_len, idx2char)

# import os

# os.chdir(r'kaggle/working')

from IPython.display import FileLink

FileLink(r'new_gates.hdf5')

FileLink(r'output.json')
