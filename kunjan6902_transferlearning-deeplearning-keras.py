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
import pandas as pd



fn = r"/kaggle/input/emotion/text_emotion.csv"

df = pd.read_csv(fn)



print(df.shape)

print(df.head(3))
df['sentiment'].unique()
indexNames = df[ df['sentiment'] == "empty" ].index

 

# Delete these row indexes from dataFrame

df = df.drop(indexNames)

print(df.shape)
df['sentiment'].nunique()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

import numpy as np



MAX_NB_WORDS = 20000

MAX_SEQUENCE_LENGTH = 300

VALIDATION_SPLIT = 0.05



processed = df["content"]



tokenizer = Tokenizer(num_words=MAX_NB_WORDS,char_level=False,oov_token=None)

tokenizer.fit_on_texts(processed)

sequences = tokenizer.texts_to_sequences(processed)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
y = df['sentiment']



#encode class values as integers

encoder = LabelEncoder()

encoder.fit(y)

encoded_Y = encoder.transform(y)



#convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)



#labels = np_utils.to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', dummy_y.shape)
# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = dummy_y[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]
print(y_val[0:10,:])
embeddings_index = {}



f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt', encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
EMBEDDING_DIM = 200



embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

        

print(embedding_matrix.shape)
from keras.layers import Embedding



embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)
from keras.layers import Input

import keras

from keras.models import Sequential

from keras.layers import Dense, InputLayer, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)



x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

#x = MaxPooling1D(30)(x)  # global max pooling

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

preds = Dense(y_val.shape[1], activation='softmax')(x)



model = keras.Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

model.summary()
history = model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs=100, batch_size=2048, verbose = 1)
from sklearn.metrics import confusion_matrix



y_pred = model.predict(x_train)

matrix_train  = confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1))

print(matrix_train)



y_pred = model.predict(x_val)
model.save("Emotion_model.h5")
import pandas as pd



df2 = pd.read_csv("/kaggle/input/twitter-product-sentiment-analysis/Twitter Product Sentiment Analysis.csv")

print(df2.head(3))
tweets = df2["tweet"]



MAX_NB_WORDS = 20000

MAX_SEQUENCE_LENGTH = 300



#tokenizer.fit_on_texts(tweets)

sequences = tokenizer.texts_to_sequences(tweets)



data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_val = data



y_pred = model.predict(x_val)

print((y_pred.shape))
k = 12

res = np.argmax(y_pred, axis = 1)

print(res.shape)
#'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',

# 'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger'

my_dict = {"0":'sadness', "1":'enthusiasm', "2":'neutral', "3":'worry', "4":'surprise', "5":'love', "6":'fun', "7":'hate', "8":'happiness',"9":'boredom', "10":'relief', "11":'anger'}



emotion_list = []



for i in range(res.shape[0]):

    emotion = my_dict[str(res[i])]

    emotion_list.append(emotion)
df2["sentiment_predict"] = emotion_list



df2.to_csv("emotion_predited_result.csv")