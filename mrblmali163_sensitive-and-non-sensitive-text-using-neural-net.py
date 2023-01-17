# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.stem.lancaster import LancasterStemmer

import os

import json

import datetime

stemmer = LancasterStemmer()





##keras



import matplotlib.pyplot as plt

np.random.seed(32)





from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.manifold import TSNE



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout

from keras.utils.np_utils import to_categorical





%matplotlib inline









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/output.csv")

data.head(10)
data.describe()
import re

def clean_text(text):

    #Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    

    #Convert to lowercase to maintain consistency

    text = text.lower()

       

    return text
data["clean_text"]=data.tweet.apply(lambda x: clean_text(x))
data.head()
STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',

              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',

              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',

              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',

              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',

              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',

              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',

              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',

              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',

              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",

              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',

              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",

              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",

              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',

              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",

              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
def remove_stop_words(txt):

    clean_txt=""

    for words in txt.split():

        if(words not in STOP_WORDS):

            clean_txt=clean_txt+words+' '

    return clean_txt

data["clean_text"]=data.clean_text.apply(lambda x: remove_stop_words(x))
data.head(10)
prepared_data=data.drop(['tweet'],axis=1)
prepared_data.head()
plt.hist(prepared_data['label'])
train_text, test_text, train_y, test_y = train_test_split(prepared_data['clean_text'],prepared_data['label'],test_size = 0.2)
print(prepared_data.shape)

print(train_text.shape)

print(test_text.shape)
train_text.head()
print(train_text.astype(str))




# get the raw text data

texts_train = train_text.astype(str)

texts_test = test_text.astype(str)



# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(nb_words=None, char_level=False)

tokenizer.fit_on_texts(texts_train)

sequences = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
print(texts_train)
sequences[0]
type(tokenizer.word_index), len(tokenizer.word_index)
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
" ".join([index_to_word[i] for i in sequences[0]])
seq_lens = [len(s) for s in sequences]

print("average length: %0.1f" % np.mean(seq_lens))

print("max length: %d" % max(seq_lens))
%matplotlib inline

import matplotlib.pyplot as plt



plt.hist(seq_lens, bins=50);
MAX_SEQUENCE_LENGTH = 51



# pad sequences with 0s

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', x_train.shape)

print('Shape of data test tensor:', x_test.shape)
y_train = train_y

y_test = test_y



y_train = to_categorical(np.asarray(y_train))

print('Shape of label tensor:', y_train.shape)
from keras.layers import Dense, Input, Flatten

from keras.layers import GlobalAveragePooling1D, Embedding

from keras.models import Model



EMBEDDING_DIM = 50

N_CLASSES = 2



# input: a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

embedded_sequences = embedding_layer(sequence_input)



average = GlobalAveragePooling1D()(embedded_sequences)

predictions = Dense(N_CLASSES, activation='sigmoid')(average)



model = Model(sequence_input, predictions)

model.compile(loss='categorical_crossentropy',

              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.4,

          nb_epoch=40, batch_size=128)
output_test = model.predict(x_test)

print("test AUC:", roc_auc_score(y_test,output_test[:,1]))
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds=precision_recall_curve(y_test,output_test[:,1])
from matplotlib import pyplot

pyplot.plot(recall, precision, marker='.')

print("precision recall curve")

pyplot.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,output_test[:,1])

pyplot.plot(fpr, tpr, marker='.')

print("roc")

pyplot.show()