import numpy as np

import pandas as pd
# importing the traning data

train_data = pd.read_csv('../input/emoji-prediction-dataset/Train.csv')

train_data.head()
# import the testing data

test_data = pd.read_csv("../input/emoji-prediction-dataset/Test.csv")

test_data.head()
# import the mappings file

mappings = pd.read_csv("../input/emoji-prediction-dataset/Mapping.csv")

mappings.head()
# print the shapes of all files

train_data.shape, test_data.shape, mappings.shape
train_length = train_data.shape[0]

test_length = test_data.shape[0]

train_length, test_length
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

stop_words[:5]
# tokenize the sentences

def tokenize(tweets):

    stop_words = stopwords.words("english")

    tokenized_tweets = []

    for tweet in tweets:

        # split all words in the tweet

        words = tweet.split(" ")

        tokenized_string = ""

        for word in words:

            # remove @handles -> useless -> no information

            if word[0] != '@' and word not in stop_words:

                # if a hashtag, remove # -> adds no new information

                if word[0] == "#":

                    word = word[1:]

                tokenized_string += word + " "

        tokenized_tweets.append(tokenized_string)

    return tokenized_tweets
# translate tweets to a sequence of numbers

def encod_tweets(tweets):

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True)

    tokenizer.fit_on_texts(tweets)

    return tokenizer, tokenizer.texts_to_sequences(tweets)
# example_str = tokenize(['This is a good day. @css #mlhlocalhost'])

# encod_str = encod_tweets(example_str)

# print(example_str)

# print(encod_str)
# apply padding to dataset and convert labels to bitmaps

def format_data(encoded_tweets, max_length, labels):

    x = pad_sequences(encoded_tweets, maxlen= max_length, padding='post')

    y = []

    for emoji in labels:

        bit_vec = np.zeros(20)

        bit_vec[emoji] = 1

        y.append(bit_vec)

    y = np.asarray(y)

    return x, y
# create weight matrix from pre trained embeddings

def create_weight_matrix(vocab, raw_embeddings):

    vocab_size = len(vocab) + 1

    weight_matrix = np.zeros((vocab_size, 300))

    for word, idx in vocab.items():

        if word in raw_embeddings:

            weight_matrix[idx] = raw_embeddings[word]

    return weight_matrix
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.wrappers import Bidirectional

from keras.layers import Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
# final model

def final_model(weight_matrix, vocab_size, max_length, x, y, epochs = 5):

    embedding_layer = Embedding(vocab_size, 300, weights=[weight_matrix], input_length=max_length, trainable=True, mask_zero=True)

    model = Sequential()

    model.add(embedding_layer)

    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))

    model.add(Bidirectional(LSTM(128, dropout=0.2)))

    model.add(Dense(20, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y, epochs = epochs, validation_split = 0.25)

    score, acc = model.evaluate(x_test, y_test)

    return model, score, acc
import math
tokenized_tweets = tokenize(train_data['TEXT'])

tokenized_tweets += tokenize(test_data['TEXT'])

max_length = math.ceil(sum([len(s.split(" ")) for s in tokenized_tweets])/len(tokenized_tweets))

tokenizer, encoded_tweets = encod_tweets(tokenized_tweets)

max_length, len(tokenized_tweets)
x, y = format_data(encoded_tweets[:train_length], max_length, train_data['Label'])

len(x), len(y)
x_test, y_test = format_data(encoded_tweets[train_length:], max_length, test_data['Label'])

len(x_test), len(y_test)
vocab = tokenizer.word_index

vocab, len(vocab)
from gensim.models.keyedvectors import KeyedVectors
# load the GloVe vectors in a dictionary:



embeddings_index = {}

f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')

for line in f:

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
weight_matrix = create_weight_matrix(vocab, embeddings_index)

len(weight_matrix)
model, score, acc = final_model(weight_matrix, len(vocab)+1, max_length, x, y, epochs = 5)

model, score, acc
model.summary()
y_pred = model.predict(x_test)

y_pred
for pred in y_pred:

    print(np.argmax(pred))
import math

from sklearn.metrics import classification_report, confusion_matrix
y_pred = np.array([np.argmax(pred) for pred in y_pred])

y_true = np.array(test_data['Label'])

print(classification_report(y_true, y_pred))
emoji_pred = [mappings[mappings['number'] == pred]['emoticons'] for pred in y_pred]

emoji_pred
for i in range(100, 150):

    test_tweet = test_data['TEXT'][i]

    pred_label = y_pred[i]

    pred_emoji = emoji_pred[i]

    print('tweet: ', test_tweet)

    print('pred emoji: ', pred_label, pred_emoji)

    print('-'*50)