# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from nltk.corpus import stopwords

import re

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"

data = pd.read_csv('../input/training.1600000.processed.noemoticon.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

data.head()

X = data.iloc[:,[5]]

Y = data.iloc[:,0]

Y[Y == 4] = 1
data.head()
# Text-preprocessing



# Missing Values

num_missing_desc = data.isnull().sum()[2]    # No. of values with msising descriptions

print('Number of missing values: ' + str(num_missing_desc))

data = data.dropna()



TAG_CLEANING_RE = "@\S+"

# Remove @tags

X['text'] = X['text'].map(lambda x: re.sub(TAG_CLEANING_RE, ' ', x))



# Smart lowercase

X['text'] = X['text'].map(lambda x: x.lower())



# Remove numbers

X['text'] = X['text'].map(lambda x: re.sub(r'\d+', ' ', x))



# Remove links

TEXT_CLEANING_RE = "https?:\S+|http?:\S|[^A-Za-z0-9]+"

X['text'] = X['text'].map(lambda x: re.sub(TEXT_CLEANING_RE, ' ', x))



# Remove Punctuation

X['text']  = X['text'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))



# Remove white spaces

X['text'] = X['text'].map(lambda x: x.strip())



# Tokenize into words

X['text'] = X['text'].map(lambda x: word_tokenize(x))

 

# Remove non alphabetic tokens

X['text'] = X['text'].map(lambda x: [word for word in x if word.isalpha()])



# Filter out stop words

stop_words = set(stopwords.words('english'))

X['text'] = X['text'].map(lambda x: [w for w in x if not w in stop_words])

    

# Word Lemmatization

lem = WordNetLemmatizer()

X['text'] = X['text'].map(lambda x: [lem.lemmatize(word,"v") for word in x])



# Turn lists back to string

X['text'] = X['text'].map(lambda x: ' '.join(x))
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("TRAIN size:", len(X_train))

print("TEST size:", len(X_train))
# Word2Vec

import gensim



# WORD2VEC 

W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10



documents = [_text.split() for _text in X_train.text] 

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)

w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
# Train Word Embeddings

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
#Test word embeddings

w2v_model.most_similar("great")
# Tokenizing

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, Dropout

from keras.utils.np_utils import to_categorical



# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 300

# This is fixed.

EMBEDDING_DIM = 300



tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train.text)

word_index = tokenizer.word_index

vocab_size = len(word_index)

print('Found %s unique tokens.' % len(word_index))



# Convert the data to padded sequences

X_train_padded = tokenizer.texts_to_sequences(X_train.text)

X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X_train_padded.shape)
# Embedding matrix for the embedding layer

embedding_matrix = np.zeros((vocab_size+1, W2V_SIZE))

for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
# Build Model

from keras.layers import Conv1D, GlobalMaxPooling1D

import keras 



model = Sequential()

model.add(Embedding(vocab_size+1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))

model.add(Dropout(0.3))

model.add(Conv1D(300,3,activation='relu'))

# we use max pooling:

model.add(GlobalMaxPooling1D())

model.add(Dense(200, activation='sigmoid'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.summary()



model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
# Training 

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]



BATCH_SIZE = 1024

history = model.fit(X_train_padded, y_train,

                    batch_size=512,

                    epochs=5,

                    validation_split=0.1,

                    verbose=1,

                    callbacks=callbacks)
# Evaluation

import matplotlib.pyplot as plt

X_test_padded = tokenizer.texts_to_sequences(X_test.text)

X_test_padded = pad_sequences(X_test_padded, maxlen=MAX_SEQUENCE_LENGTH)

score = model.evaluate(X_test_padded, y_test, batch_size=512)

print("ACCURACY:",score[1])

print("LOSS:",score[0])



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
model.save('/kaggle/working/Sentiment_CNN_model.h5')
def predict(text, include_neutral=True):

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]

    return {"score": float(score)}  







predict("@Nintendo I love your games!")