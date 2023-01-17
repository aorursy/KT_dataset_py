# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

#!pip install pymystem3

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
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU

from tensorflow.keras import utils

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import utils

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

%matplotlib inline 
# Максимальное количество слов 

num_words = 10000

# Максимальная длина платежки

max_news_len = 250

# Количество классов новостей

nb_classes = 3
train_data = pd.read_csv('/kaggle/input/fsspdata/train.csv')
test_data = pd.read_csv('/kaggle/input/fsspdata/test.csv')
train_data.head()
test_data.head()
#train_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'],axis=1,inplace=True)

#train_data.info()
train_data.head()
train_data.shape
test_data.shape
def remove_punc(text):

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub( '[№!@#$]', '', text)

    return text

train_data['Purpos'] = train_data['Purpos'].apply(lambda x: remove_punc(x))
test_data['Purpos'] = test_data['Purpos'].apply(lambda x: remove_punc(x))
def data_cleaner(text):        

    lower_case = text.lower()

    tokens=word_tokenize(lower_case)

    return (" ".join(tokens)).strip()

train_data['Purpos'] = train_data['Purpos'].apply(lambda x: data_cleaner(x))

test_data['Purpos'] = test_data['Purpos'].apply(lambda x: data_cleaner(x))
import string

def remove_punctuation(text):

    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])



def remove_numbers(text):

    return ''.join([i if not i.isdigit() else ' ' for i in text])



import re

def remove_multiple_spaces(text):

	return re.sub(r'\s+', ' ', text, flags=re.I)
train_data['Purpos'] = train_data['Purpos'].apply(lambda x: remove_punctuation(x))

train_data['Purpos'] = train_data['Purpos'].apply(lambda x: remove_numbers(x))

train_data['Purpos'] = train_data['Purpos'].apply(lambda x: remove_multiple_spaces(x))
test_data['Purpos'] = test_data['Purpos'].apply(lambda x: remove_punctuation(x))

test_data['Purpos'] = test_data['Purpos'].apply(lambda x: remove_numbers(x))

test_data['Purpos'] = test_data['Purpos'].apply(lambda x: remove_multiple_spaces(x))
train_data.head()
russian_stopwords = stopwords.words("russian")

russian_stopwords.extend(['…', '«', '»', '...', 'т.д.','ндс'])



def remove_stopwords (text):        

    list1=[word for word in text.split() if word not in russian_stopwords]

    #print(list1)

    return " ".join(list1)
train_data['Purpos'] = train_data['Purpos'].apply(lambda x: remove_stopwords(x))

test_data['Purpos'] = test_data['Purpos'].apply(lambda x: remove_stopwords(x))
lemm_texts_list = []

for text in tqdm(train_data['Purpos']):

    #print(text)

    try:

        word_list = nltk.word_tokenize(text)

        tokens = [token for token in word_list if token != ' ']

        text = " ".join(tokens)

        lemm_texts_list.append(text)

    except Exception as e:

        print(e)

    

train_data['text_lemm'] = lemm_texts_list

train_data['text_lemm'][50:]
test_data.shape
lemm_texts_list_test = []

for text in tqdm(test_data['Purpos']):

    #print(text)

    try:

        word_list = nltk.word_tokenize(text)

        tokens = [token for token in word_list if token != ' ']

        text = " ".join(tokens)

        lemm_texts_list_test.append(text)

    except Exception as e:

        print(e)

    

test_data['text_lemm'] = lemm_texts_list_test
test_data['text_lemm'][50:]

y_train = utils.to_categorical(train_data['Cat'] - 1, nb_classes)
y_test = utils.to_categorical(test_data['Cat'] - 1, nb_classes)
y_train
y_test
tokenizer = Tokenizer(num_words=num_words)
purpose = train_data['text_lemm']


tokenizer.fit_on_texts(purpose)
tokenizer.word_index
sequences = tokenizer.texts_to_sequences(purpose)
index = 1

print(purpose[index])

print(sequences[index])
index = 1

print(purpose[index])

print(sequences[index])
x_train = pad_sequences(sequences, maxlen=max_news_len)
x_train[:5]
model_cnn = Sequential()

model_cnn.add(Embedding(num_words, 32, input_length=max_news_len))

model_cnn.add(Conv1D(250, 3, padding='valid', activation='relu'))

model_cnn.add(GlobalMaxPooling1D())

model_cnn.add(Dense(128, activation='relu'))

model_cnn.add(Dense(3, activation='softmax'))
model_cnn.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model_cnn.summary()
model_cnn_save_path = 'best_model_cnn.h5'

checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path, 

                                      monitor='val_accuracy',

                                      save_best_only=True,

                                      verbose=1)
history_cnn = model_cnn.fit(x_train, 

                            y_train, 

                            epochs=5,

                            batch_size=128,

                            validation_split=0.1,

                            callbacks=[checkpoint_callback_cnn])
plt.plot(history_cnn.history['accuracy'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history_cnn.history['val_accuracy'], 

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()
model_lstm = Sequential()

model_lstm.add(Embedding(num_words, 32, input_length=max_news_len))

model_lstm.add(LSTM(16))

model_lstm.add(Dense(3, activation='softmax'))
model_lstm.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model_lstm.summary()
model_lstm_save_path = 'best_model_lstm.h5'

checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 

                                      monitor='val_accuracy',

                                      save_best_only=True,

                                      verbose=1)
history_lstm = model_lstm.fit(x_train, 

                              y_train, 

                              epochs=5,

                              batch_size=128,

                              validation_split=0.1,

                              callbacks=[checkpoint_callback_lstm])
plt.plot(history_lstm.history['accuracy'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history_lstm.history['val_accuracy'], 

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()
model_gru = Sequential()

model_gru.add(Embedding(num_words, 32, input_length=max_news_len))

model_gru.add(GRU(16))

model_gru.add(Dense(3, activation='softmax'))
model_gru.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model_gru.summary()
model_gru_save_path = 'best_model_gru.h5'

checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path, 

                                      monitor='val_accuracy',

                                      save_best_only=True,

                                      verbose=1)
history_gru = model_gru.fit(x_train, 

                              y_train, 

                              epochs=5,

                              batch_size=128,

                              validation_split=0.1,

                              callbacks=[checkpoint_callback_gru])
plt.plot(history_gru.history['accuracy'], 

         label='Доля верных ответов на обучающем наборе')

plt.plot(history_gru.history['val_accuracy'], 

         label='Доля верных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля верных ответов')

plt.legend()

plt.show()
test_sequences = tokenizer.texts_to_sequences(test_data['text_lemm'])
x_test = pad_sequences(test_sequences, maxlen=max_news_len)
x_test[:5]
model_cnn.load_weights(model_cnn_save_path)
model_cnn.evaluate(x_test, y_test, verbose=1)
model_lstm.load_weights(model_lstm_save_path)
model_lstm.evaluate(x_test, y_test, verbose=1)
model_gru.load_weights(model_gru_save_path)
model_gru.evaluate(x_test, y_test, verbose=1)
examples = ['Заработная плата за июль. Питин Михаил Юрьевич',

           'Зарплата за июль. Кукин Михаил Юрьевич',

           'Оплата алиментов']



d = {'text': examples}

df = pd.DataFrame(data=d)

print(df)





  

df['text'] = df['text'].apply(lambda x: remove_punc(x))

df['text'] = df['text'].apply(lambda x: data_cleaner(x))

#examples = remove_punc(examples)

#examples = data_cleaner(examples)
df['text'] = df['text'].apply(lambda x: remove_punctuation(x))

df['text'] = df['text'].apply(lambda x: remove_numbers(x))

df['text'] = df['text'].apply(lambda x: remove_multiple_spaces(x))

#examples = remove_punctuation(examples)

#examples = remove_numbers(examples)

#examples = remove_multiple_spaces(examples)
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

#examples = remove_stopwords(examples)
lemm_texts_list_ex = []

for text in tqdm(df['text']):

    #print(text)

    try:

        word_list = nltk.word_tokenize(text)

        tokens = [token for token in word_list if token != ' ']

        text = " ".join(tokens)

        lemm_texts_list_ex.append(text)

    except Exception as e:

        print(e)

    

df['text_lemm'] = lemm_texts_list_ex



print(df['text_lemm'])
ex_sequences = tokenizer.texts_to_sequences(df['text_lemm'])

ex = pad_sequences(ex_sequences)









print(ex)

model_cnn.predict(ex)