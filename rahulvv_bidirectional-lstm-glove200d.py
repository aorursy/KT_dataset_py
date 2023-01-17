# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Loading word vectors...')

word2vec = {}

with open(os.path.join('../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'), encoding = "utf-8") as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

    for line in f:

        values = line.split() #split at space

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32') #numpy.asarray()function is used when we want to convert input to an array.

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
import pandas as pd

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_df.head()
train_df.isnull().sum()
import pandas as pd

import numpy as np



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import string



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, LSTM, Embedding

from tensorflow.keras.models import Sequential

import os



nltk.download('stopwords')

nltk.download('wordnet')
print(train_df.shape)

print(test_df.shape)
train_sentences = train_df.text.values

train_labels = train_df.target.values

test_sentences = test_df.text.values
import string

import re



#remove urls

def remove_urls(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)

  

#remove html tags

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



# splitting the text

def split_text(text):

    text = text.split()

    return text



 # making lower case words

def lower(text):

    text = [word.lower() for word in text]

    return str(text)



#remove punct

def remove_punct(text):

    text = ''.join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', str(text))

    return text



#remove stopwords

def remove_stopwords(text):

    pattern = re.compile(r'\b('+r'|'.join(stopwords.words('english')) + r')\b\s*')

    text = pattern.sub(' ', text)

    return text



#lemmatization

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    text = lemmatizer.lemmatize(text)

    return text
def final(text):

    t0 = remove_urls(text)

    t1 = remove_html(t0)

    t2 = split_text(t1)

    t3 = lower(t2)

    t4 = remove_punct(t3)

    t5 = remove_stopwords(t4)

    t6 = lemmatize_words(t5)

    return t6
training_sentences = []

for i in range(len(train_sentences)):

    data = final(train_sentences[i])

    training_sentences.append(data)



testing_sentences = []

for i in range(len(test_sentences)):

    data = final(test_sentences[i])

    testing_sentences.append(data)
num_words = 20000

embedding_dim = 200

max_length = 32

trunc_type='post'

padding_type='post'

oov_tok = 'OOV'



tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

train_sequences = tokenizer.texts_to_sequences(training_sentences)



word_index = tokenizer.word_index

padded_train = pad_sequences(sequences=train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



print('Total unique tokens generated: ',len(word_index))

print('Shape of padded train tensor: ', padded_train.shape)



#tokenizer = Tokenizer(num_words=num_words, oov_token=oov_tok)

#tokenizer.fit_on_texts(testing_sentences)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)

padded_test = pad_sequences(sequences=test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



print('Shape of padded test tensor: ', padded_test.shape)
num_words = min(20000, len(word_index)+1)

embedding_matrix = np.zeros((num_words, embedding_dim))



embeddings = []

for word, i in word_index.items():

    if i<20000:

        embeddings = word2vec.get(word)

        if embeddings is not None:

            embedding_matrix[i] = embeddings
from tensorflow.keras.callbacks import ModelCheckpoint

model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=num_words,

                                                       output_dim = embedding_dim,

                                                       weights=[embedding_matrix],

                                                       input_length=max_length,

                                                       trainable=False),

                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),

                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),

                             tf.keras.layers.Flatten(),

                             tf.keras.layers.Dense(1, activation='sigmoid')])



model.compile(loss='binary_crossentropy',

              optimizer=tf.keras.optimizers.Adam(lr=0.01),

              metrics=['accuracy'])



#checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)





model.summary()
history=model.fit(padded_train,

                  train_labels,

                  batch_size=128,

                  epochs=15)
pred = model.predict(padded_test)

pred = (np.round(pred)).astype(int)

pred = pred.reshape(3263)
'''from matplotlib import pyplot

pyplot.figure(figsize=(12,7))

pyplot.plot(history.history['loss'])

pyplot.plot(history.history['val_loss'])

pyplot.title('model train vs validation loss')

pyplot.ylabel('loss')

pyplot.xlabel('epoch')

pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.show()

pyplot.figure(figsize=(12,7))

pyplot.plot(history.history['accuracy'])

pyplot.plot(history.history['val_accuracy'])

pyplot.title('model train vs validation loss')

pyplot.ylabel('accuracy')

pyplot.xlabel('epoch')

pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.show()'''
my_submission = pd.DataFrame({'id': test_df.id, 'target': pred})

my_submission.to_csv('submission.csv', index=False)