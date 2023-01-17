from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import pandas as pd

import numpy as np

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

import pickle

import re

from keras.preprocessing import sequence,text

from keras.models import load_model

import re

from nltk.corpus import stopwords

import gensim

from gensim.models import KeyedVectors

from gensim.test.utils import datapath

import nltk

import string
df = pd.read_csv("/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep='\t')

df.head()
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^a-z ]')

STOPWORDS = set(stopwords.words('english'))

stemmer = nltk.stem.WordNetLemmatizer()



def text_prepare(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = text.lower()# lowercase text

#     text = " ".join([contractions.fix(t) for t in text.split(' ')])

    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)# replace REPLACE_BY_SPACE_RE symbols by space in text

    text = re.sub(BAD_SYMBOLS_RE, '', text)# delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join([word.strip(string.punctuation) for word in text.split(' ')])

    text = ' '.join([t for t in text.split(" ") if len(t) > 2])

    text = ' '.join([stemmer.lemmatize(x) for x in text.split() if x not in STOPWORDS])# delete stopwords from text

    return text
df["Sentiment"].value_counts()
print("Mean of word count in test set",np.mean(df['Phrase'].apply(lambda x: len(x.split()))))

print("Max of word count in test set",np.max(df['Phrase'].apply(lambda x: len(x.split()))))

print("Min of word count in test set",np.min(df['Phrase'].apply(lambda x: len(x.split()))))
results = set()

df['Phrase'].str.lower().str.split().apply(results.update)

print("Unique  words in phrases", len(results))
x = df["Phrase"].apply(text_prepare)

y = df["Sentiment"]



# trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.33, random_state=42)
max_features = 17000

max_words = 55



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(x)

x_train = tokenizer.texts_to_sequences(x)

# X_val = tokenizer.texts_to_sequences(X_val_text)

# X_test = tokenizer.texts_to_sequences(test_text)



x1 = sequence.pad_sequences(x_train, maxlen=max_words)

# X_val = sequence.pad_sequences(X_val, maxlen=max_words)

# X_test = sequence.pad_sequences(X_test, maxlen=max_words)



# saving

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# # loading

# with open('tokenizer.pickle', 'rb') as handle:

#     tokenizer = pickle.load(handle)

target = to_categorical(y)
model=Sequential()

model.add(Embedding(max_features,250,mask_zero=True))

model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))

model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model.summary()
%%time

history=model.fit(x1, target, epochs=5, batch_size=50, verbose=1)


# model.save('sentiment_model.h5')  # creates a HDF5 file 'my_model.h5'

# del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

# model = load_model('sentiment_model.h5')
df1 = pd.read_csv("/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")

df1.head()
testx = df1["Phrase"].apply(text_prepare)



testx1 = tokenizer.texts_to_sequences(testx)

# X_val = tokenizer.texts_to_sequences(X_val_text)

# X_test = tokenizer.texts_to_sequences(test_text)



testx1 = sequence.pad_sequences(testx1, maxlen=max_words)

out1 = model.predict_classes(testx1)

out1
x4 = pd.DataFrame({'PhraseId':np.array(df1["PhraseId"]), 'Sentiment':out1})

# x4["Sentiment"] = x4["Sentiment"].apply(lambda x:int(round(x)))

x4.to_csv("output.csv", index=False)