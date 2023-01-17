# Keras

from keras.models import Sequential

from keras.layers import Bidirectional, LSTM, Dense, Embedding, SpatialDropout1D

from keras.optimizers import adam

from keras.callbacks import EarlyStopping

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder

import keras



# Regular Expression

import re



# NLTK

from nltk.tokenize import word_tokenize

from nltk import FreqDist

from nltk.stem import WordNetLemmatizer



# EDA

from string import punctuation

import pandas as pd

import numpy as np
# Data Load

df = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="	")

df.head()
len(df)
df.isnull().sum()
df['Sentiment'].value_counts()
# Preprocessing

df['Phrase'] = df['Phrase'].apply(lambda x: x.lower())

df['Phrase'] = df['Phrase'].apply((lambda x: re.sub('[^A-z\s]','',x)))
lemma=WordNetLemmatizer()

def clean_text(text):

    text_corpus=[]

    for i in range(0,len(text)):

        review = str(text[i])

        review = [lemma.lemmatize(w) for w in word_tokenize(str(review))]

        review = ' '.join(review)

        text_corpus.append(review)

    return text_corpus
#df['Phrase'] = df['Phrase'].map(lambda x : x if len(x.split(" ")) > 1 else None)
df['clean_text'] = clean_text(df['Phrase'].values)

df.head()
# Total Words

aa = ' '.join(list(df['clean_text']))

aa = list(set(aa.split(" ")))

len(aa)
from sklearn.utils import shuffle
df = shuffle(df)

df.head()
# Tokenizer

vocabulary_size = len(aa)

tokenizer = Tokenizer(num_words=vocabulary_size, split=' ')

tokenizer.fit_on_texts(df['clean_text'].values)

sequences = tokenizer.texts_to_sequences(df['clean_text'].values)

data = pad_sequences(sequences)#, maxlen=45)
from keras.utils.np_utils import to_categorical
# Encoder

encoder = LabelEncoder()

encoder = encoder.fit_transform(df['Sentiment'])

target = to_categorical(encoder)
data.shape, target.shape
from keras.backend import zeros
embeddings_index = dict()

f = open('../input/glove6b300dtxt/glove.6B.300d.txt', encoding='utf-8')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()
embedding_matrix = np.zeros((vocabulary_size, 300))

for word, index in tokenizer.word_index.items():

    if index > vocabulary_size - 1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector
from keras.layers import GRU, Dropout
# Model

model = Sequential()

model.add(Embedding(vocabulary_size, 300, input_length = data.shape[1], weights = [embedding_matrix], trainable=True))

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(LSTM(256, return_sequences=True)))

model.add(Bidirectional(GRU(256)))

model.add(Dropout(0.5))

model.add(Dense(target.shape[1], activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
early_stopping_filter = EarlyStopping(monitor='val_loss', patience=2)
#model.fit(data, target, validation_split=0.1, epochs=4, callbacks=[early_stopping_filter], batch_size=256)

model.fit(data, target, epochs=4, callbacks=[early_stopping_filter], batch_size=256)
testdf = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="	")

testdf.head()
testdf['Phrase'] = testdf['Phrase'].apply(lambda x: x.lower())

testdf['Phrase'] = testdf['Phrase'].apply((lambda x: re.sub('[^A-z\s]','',x)))

testdf['clean_test'] = clean_text(testdf['Phrase'].values)
test_sequences = tokenizer.texts_to_sequences(testdf['clean_test'].values)

test_data = pad_sequences(test_sequences, maxlen=data.shape[1])
y_pred = model.predict_classes(test_data, verbose=1)
submissiondf = pd.DataFrame({'PhraseId': testdf['PhraseId'], 'Sentiment': y_pred})

submissiondf.head()
submissiondf.to_csv("sampleSubmission.csv", index=False)