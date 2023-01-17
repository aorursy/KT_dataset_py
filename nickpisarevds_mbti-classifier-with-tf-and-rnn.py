import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

from matplotlib.pyplot import *

import seaborn as sns

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('punkt')



%matplotlib inline

from nltk import tokenize



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from keras.engine.topology import Layer

from keras import initializers as initializers, regularizers, constraints

from keras.utils.np_utils import to_categorical

from keras import optimizers

from keras.models import Model
data=pd.read_csv('../input/mbti-type/mbti_1.csv')

data.head()
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords 

from nltk import word_tokenize



stemmer = PorterStemmer()

lemmatiser = WordNetLemmatizer() 

cachedStopWords = stopwords.words("english")



def cleaning_data(data, remove_stop_words=True):

    list_posts = []

    i=0   

    for row in data.iterrows():

        posts = row[1].posts

        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts) #remove urls

        temp = re.sub("[^a-zA-Z.]", " ", temp) #remove all punctuations except fullstops.

        temp = re.sub(' +', ' ', temp).lower() 

        temp=re.sub(r'\.+', ".", temp) #remove multiple fullstops.

        if remove_stop_words:

            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])

        else:

            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        list_posts.append(temp)



    text = np.array(list_posts)

    return text
clean_text = cleaning_data(data, remove_stop_words=True)

data['clean_text']=clean_text

data = data[['clean_text', 'type']]

data.head()
types=data['type']

text=data['clean_text']

tps=data.groupby('type')

print("total types:",tps.ngroups)

print(tps.size())
max_len=200   # maximum words in a sentence

VAL_SPLIT = 0.2



tokenizer = Tokenizer()

tokenizer.fit_on_texts(text)

max_features = len(tokenizer.word_index) + 1 # maximum number of unique words





input_sequences = []

for line in (data):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)
max_seq_length = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_seq_length, padding='pre'))



xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=max_features, dtype='float64')



x_val = xs * VAL_SPLIT

y_val = ys * VAL_SPLIT
model = Sequential()

model.add(Embedding(max_features, 64, input_length = max_seq_length - 1))

model.add(tf.keras.layers.Conv1D(32, (1), padding='same', activation='relu'))

model.add(tf.keras.layers.Conv1D(32, (1), activation='relu'))

model.add(tf.keras.layers.Dropout(0.5)) 



model.add(tf.keras.layers.Conv1D(64, (1), padding='same', activation='relu'))

model.add(tf.keras.layers.Conv1D(64, (1), activation='relu'))

model.add(tf.keras.layers.Dropout(0.5)) 



model.add(Bidirectional(LSTM(64)))

model.add(Dense(max_features, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

history = model.fit(xs, ys, epochs = 500, validation_data=(x_val, y_val), verbose = 1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
model.save('mbti_rnn.h5')