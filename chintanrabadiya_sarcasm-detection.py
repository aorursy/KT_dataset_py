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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize

from bs4 import BeautifulSoup

import re,string,unicodedata

from keras.preprocessing import text, sequence

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

import keras

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU

import tensorflow as tf
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)

df['headline']
df.isna().sum() # Checking for NaN values
sentences = []

labels = []

for item in df['headline']:

    sentences.append(item)

for item in df['is_sarcastic']:

    labels.append(item)
vocab_size =1000

embedding_dim = 20

max_length = 150

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size = 20000
training_sentences = sentences[0:training_size]

testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]

testing_labels = labels[training_size:]
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)



word_index = tokenizer.word_index



training_seq = tokenizer.texts_to_sequences(training_sentences)

training_pad = pad_sequences(training_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)



testing_seq = tokenizer.texts_to_sequences(testing_sentences)

testing_pad = pad_sequences(testing_seq, maxlen=max_length, padding= padding_type, truncating=trunc_type)
import numpy as np

training_pad = np.array(training_pad)

training_labels = np.array(training_labels)

testing_pad = np.array(testing_pad)

testing_labels = np.array(testing_labels)
model = Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(optimizer='adam',

             loss = 'binary_crossentropy',

             metrics=['accuracy'])

model.summary()
num_epoch=30

history = model.fit(training_pad,training_labels, epochs=num_epoch, validation_data=(testing_pad,testing_labels), verbose=2)
pred = model.predict_classes(testing_pad)
pred

print(classification_report(testing_labels, pred, target_names = ['Not Sarcastic','Sarcastic']))