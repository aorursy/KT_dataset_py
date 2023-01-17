import pandas as pd

import numpy as np

import re



from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split 

from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers
data_IMDB = pd.read_csv('../input/imdb_master.csv',encoding="latin-1")

data_IMDB.head()
data_IMDB.drop(['Unnamed: 0', 'file', 'type'], axis = 1, inplace = True)

data_IMDB = data_IMDB[data_IMDB.label != 'unsup']

data_IMDB['label'] = data_IMDB['label'].map({'pos': 1, 'neg': 0})

data_IMDB.head()
stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()





def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



data_IMDB['Processed_Reviews'] = data_IMDB.review.apply(lambda x: clean_text(x))



data_IMDB.head()
data_IMDB.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(data_IMDB['Processed_Reviews'])

list_tokenized_train = tokenizer.texts_to_sequences(data_IMDB['Processed_Reviews'])



maxlen = 130

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

y = data_IMDB['label']



embed_size = 128

model = Sequential()

model.add(Embedding(max_features, embed_size))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 3

model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)