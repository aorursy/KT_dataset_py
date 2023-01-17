import numpy as np

import pandas as pd

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_url = '/kaggle/input/imdb-review-dataset/imdb_master.csv'

data = pd.read_csv(data_url, encoding='latin-1')

data.head(10)
data.shape
data['type'].value_counts()
data['label'].value_counts()
pd.crosstab(data['label'], data['type'])
data = data[data.label != 'unsup']
data = data.drop(['Unnamed: 0', 'file', 'type'], axis=1)
data.label = data.label.map({'neg':0, 'pos':1})
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, LSTM, Dropout, Dense

from keras.models import Model

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))



def preprocess_text(text):

    text = text.lower()

    text = ''.join([char for char in text if char not in string.punctuation])

    tokens = text.split(' ')

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in stop_words]

    return tokens
data['preprocessed_reviews'] = data.review.apply(preprocess_text)
data.head()
tokenizer = Tokenizer(50000)

tokenizer.fit_on_texts(data['preprocessed_reviews'])
sequences = tokenizer.texts_to_sequences(data['preprocessed_reviews'])
label = data['label']
print("Average length of sequences:", np.mean([len(seq) for seq in sequences]))
maxlen = 160

padded_sequences = pad_sequences(sequences, maxlen=maxlen)
vocab_length = len(tokenizer.word_index)

inputs = Input(shape=(160,))

X = Embedding(vocab_length+1, 128)(inputs)

X = LSTM(32)(X)

X = Dense(32)(X)

X = Dropout(0.3)(X)

outputs = Dense(1, activation='sigmoid')(X)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
reducelr = ReduceLROnPlateau(patience=2, verbose=1)

earlystopping = EarlyStopping(patience=3, verbose=1)
batch_size = 100

epochs = 5

hist = model.fit(padded_sequences, label, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[reducelr, earlystopping])