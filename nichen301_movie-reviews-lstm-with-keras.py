import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", delimiter='\t')

train.head()
train_imdb = pd.read_csv("../input/imdb-review-dataset/imdb_master.csv", encoding="latin-1", index_col=0)

train_imdb.head()
train_imdb = train_imdb[train_imdb.label != 'unsup']

train_imdb["sentiment"] = train_imdb.label.map({"neg": 0, "pos": 1})

train_imdb.drop(["type", "file", "label"], axis=1, inplace=True)
train_enhanced = pd.concat([train, train_imdb], ignore_index=True)



X_train = train_enhanced.review

y_train = train_enhanced.sentiment
import re

from nltk.corpus import stopwords



stopwords_eng = set(stopwords.words("english"))



def clean_text(text):

    text = text.lower() # convert to lowercase

    text = re.sub("[^a-z]", " ", text)

    words = [word for word in text.split() if word not in stopwords_eng]

    text = " ".join(words)

    return text



X_train = X_train.map(clean_text)

X_train.head()
from keras.preprocessing.text import Tokenizer

num_words = 6000

tokenizer = Tokenizer(num_words=num_words)



tokenizer.fit_on_texts(X_train)



X_train_seq = pd.Series(tokenizer.texts_to_sequences(X_train))

X_train_seq.head()
X_train_len = X_train_seq.map(lambda ls: len(ls))

X_train_len.describe()
from keras.preprocessing.sequence import pad_sequences



X_train_pad = pad_sequences(X_train_seq, maxlen=512)
from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D, Dropout



model = Sequential()



model.add(Embedding(input_dim=num_words, output_dim=64))

model.add(LSTM(32, return_sequences=True))

model.add(GlobalMaxPool1D())

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])



batch_size = 32

epochs = 2

validation_split = 0.01

model.fit(x=X_train_pad, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv", delimiter="\t")

test.head()
X_test = test.review.map(clean_text)

X_test.head()
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_test_pad = pad_sequences(X_test_seq, maxlen=512)
pred = model.predict(x=X_test_pad)

y_pred = (pred >= 0.5) * 1
results = pd.DataFrame({"id": test.id, "sentiment": y_pred.flatten()})

results.to_csv("submission.csv", index=False)