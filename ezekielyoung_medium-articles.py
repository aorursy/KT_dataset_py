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
DIR = "../input/medium-articles-dataset/medium_data.csv"
import numpy

import matplotlib.pyplot as plt

import pandas as pd

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np





import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer







from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder






import re

df = pd.read_csv(DIR)
df.head()
(df.subtitle[0] == df.subtitle[0] )

df.describe()
df = df.drop(['image', 'id', 'url'], axis=1)

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



def clean(text):

    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()

    return text

df.title = df.title.apply(lambda x: clean(x))

df.subtitle = df.subtitle.apply(lambda x: clean(x))



# df.responses  = df.responses.apply(lambda x: int(x))

def cleaner(num):

    if (num > 500):

        return 500

    else:

        return num

df.claps = df.claps.apply(lambda x: cleaner(x))
print(df.describe())

df.hist(bins =25)
TRAIN_SIZE = 0.8

MAX_NB_WORDS = 100000

MAX_SEQUENCE_LENGTH = 20



train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE,

                                         random_state=7) # Splits Dataset into Training and Testing set

print("Train Data size:", len(train_data))

print("Test Data size", len(test_data))
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_data.title)



word_index = tokenizer.word_index

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary Size :", vocab_size)



from keras.preprocessing.sequence import pad_sequences



x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.title),

                        maxlen = MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.title),

                       maxlen = MAX_SEQUENCE_LENGTH)



subtitle_train = pad_sequences(tokenizer.texts_to_sequences(train_data.subtitle),

                        maxlen = MAX_SEQUENCE_LENGTH)

subtitle_test = pad_sequences(tokenizer.texts_to_sequences(test_data.subtitle),

                       maxlen = MAX_SEQUENCE_LENGTH)



responses_train = train_data.responses

responses_test = test_data.responses





reading_time_train = train_data.reading_time

reading_time_test = test_data.reading_time





y_train = train_data.claps

y_test = test_data.claps



print("Training X Shape:",x_train.shape)

print("Testing X Shape:",x_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)
GLOVE_DIR = '../input/glove1/glove.6B.300d.txt'





embeddings_index = {}

f = open(GLOVE_DIR)

print('Loading GloVe from:', GLOVE_DIR,'...', end='')

for line in f:

    values = line.split()

    word = values[0]

    embeddings_index[word] = np.asarray(values[1:], dtype='float32')

f.close()

print("Done.\n Proceeding with Embedding Matrix...", end="")



embedding_matrix = np.random.random((len(word_index) + 1, 300))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

print(" Completed!")
#1

# from tensorflow.keras.layers import *





# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# embedding_layer = Embedding(len(word_index) + 1,

#                            300,

#                            weights = [embedding_matrix],

#                            input_length = MAX_SEQUENCE_LENGTH,

#                            trainable=False,# prevent re-training the glove vector

#                            name = 'embeddings')

# embedded_sequences = embedding_layer(sequence_input)

# x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)

# x = GlobalMaxPool1D()(x)

# x = Dropout(0.1)(x)

# x = Dense(50, activation="relu")(x)

# x = Dropout(0.1)(x)

# preds = Dense(1)(x)





# ####



# from tensorflow.keras.models import Model



# model = Model(sequence_input, preds)

# model.compile(loss='mean_squared_error', optimizer='adam',

#              metrics = ['mae'])
# model = keras.Sequential([

#       norm,

#       layers.Dense(64, activation='relu'),

#       layers.Dense(64, activation='relu'),

#       layers.Dense(1)

#   ])



#   model.compile(loss='mean_absolute_error',

#                 optimizer=tf.keras.optimizers.Adam(0.001))
#1

from tensorflow.keras.layers import *



# Title

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(len(word_index) + 1,

                           300,

                           weights = [embedding_matrix],

                           input_length = MAX_SEQUENCE_LENGTH,

                           trainable=False,# prevent re-training the glove vector

                           name = 'embeddings')

embedded_sequences = embedding_layer(sequence_input)

x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)

# x = Conv1D(64, 5, activation='relu')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

preds = Dense(10)(x)



#Subtitle

sequence_input2= Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedded_sequences2 = embedding_layer(sequence_input2)

x2 = LSTM(60, return_sequences=True,name='lstm_layer2')(embedded_sequences2)

# x = Conv1D(64, 5, activation='relu')(x)

x2 = GlobalMaxPool1D()(x2)

x2 = Dropout(0.1)(x2)

x2 = Dense(50, activation="relu")(x2)

x2 = Dropout(0.1)(x2)

preds2 = Dense(10)(x2)



#reading_time

input_y = Input(shape = (1,), dtype='int32')

y = Dense(2, activation="relu")(input_y)



#Supposedly responses but there's something wrong with the data

# input_z = Input(shape = (1,), dtype='int32')

# z = Dense(2, activation="relu")(input_z)



# concat = Concatenate()([preds, y, z])



concat = Concatenate()([preds, preds2, y])



output = Dense(1)(concat)



####



from tensorflow.keras.models import Model



# model = Model(inputs = [sequence_input, input_y, input_z], outputs = [output])

model = Model(inputs = [sequence_input, sequence_input2, input_y], outputs = [output])



model.compile(loss='mean_absolute_error', optimizer='adam',

             metrics = ['mae'])

# sad



######



print('Training progress:')

# history = model.fit([x_train, responses_train,reading_time_train], y_train, epochs = 50, batch_size=64, validation_data=([x_test, responses_test,reading_time_test], y_test))



history = model.fit([x_train,subtitle_train, reading_time_train], y_train, epochs = 15, batch_size=64, validation_data=([x_test,subtitle_test, reading_time_test], y_test))



# [x_test, responses_test,reading_time_test]

mae = history.history['mae']

val_mae = history.history['val_mae']

epochs = range(1, len(mae)+1)



plt.plot(epochs, mae, label='Training mae')

plt.plot(epochs, val_mae, label='Validation MAE')

plt.title('Training and validation MAE')

plt.ylabel('MAE/val_MAE')

plt.xlabel('Epochs')

plt.legend()

plt.show();
# #2 

# from tensorflow.keras.layers import *





# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



# embedding_layer = Embedding(len(word_index) + 1,

#                            300,

#                            weights = [embedding_matrix],

#                            input_length = MAX_SEQUENCE_LENGTH,

#                            trainable=False,# prevent re-training the glove vector

#                            name = 'embeddings')

# embedded_sequences = embedding_layer(sequence_input)

# x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)

# x = Dense(64, activation="relu")(x)

# x = Dense(64, activation="relu")(x)

# x = Dense(64, activation="relu")(x)



# preds = Dense(10)(x)







# ####



# from tensorflow.keras.models import Model



# model = Model(sequence_input, preds)

# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001),

#              metrics = ['mae'])







# ######



# print('Training progress:')

# history = model.fit(x_train, y_train, epochs = 50, batch_size=64, validation_data=(x_test, y_test))



# mae = history.history['mae']

# val_mae = history.history['val_mae']

# epochs = range(1, len(mae)+1)



# plt.plot(epochs, mae, label='Training mae')

# plt.plot(epochs, val_mae, label='Validation MAE')

# plt.title('Training and validation MAE')

# plt.ylabel('MAE/val_MAE')

# plt.xlabel('Epochs')

# plt.legend()

# plt.show();
tf.keras.utils.plot_model(model)

test_predictions = model.predict([x_test,subtitle_test, reading_time_test]).flatten()



# a = plt.axes(aspect='equal')

plt.scatter(y_test, test_predictions)

plt.xlabel('True Values [Claps]')

plt.ylabel('Predictions [Claps]')
#1

mae = history.history['mae']

val_mae = history.history['val_mae']

epochs = range(1, len(mae)+1)



plt.plot(epochs, mae, label='Training mae')

plt.plot(epochs, val_mae, label='Validation MAE')

plt.title('Training and validation MAE')

plt.ylabel('MAE/val_MAE')

plt.xlabel('Epochs')

plt.legend()

plt.show();
# print(x_test.shape)

# print(y_test.shape)

# test_predictions = model.predict(x_test)

# print(test_predictions.flatten().shape)

# print(test_predictions.shape)
