# Import modules

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
nRowsRead = 10



df = pd.read_csv("../input/final_data.csv", delimiter=',', nrows = nRowsRead)

df = df.sample(frac=1)

df.head(3)
df.columns
# Loading the data

df = pd.read_csv("../input/final_data.csv", delimiter=',', encoding = "ISO-8859-1", nrows = None)

# Shuffling the data

df = df.sample(frac=1)

# Selecting the interesting columns

df = df[["reviews.doRecommend", "reviews.text", "reviews.title"]]
df.head()
df.info()
df = df.dropna()

df.info()
df["reviews.doRecommend"].value_counts()
df["reviews.doRecommend"].astype(float).hist(figsize=(8,5))
df.rename(columns={'reviews.title':'title','reviews.doRecommend':'doRecommend'}, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df, df["doRecommend"]):

    train_set = df.iloc[train_index]

    test_set = df.iloc[test_index]
print(train_set["doRecommend"].value_counts()/len(train_set))

print(test_set["doRecommend"].value_counts()/len(test_set))
train_set = train_set.drop(columns = ["reviews.text"])

test_set = test_set.drop(columns = ["reviews.text"])



train_set["title"] = train_set["title"].astype(str)

test_set["title"] = test_set["title"].astype(str)



train_set["doRecommend"] = train_set["doRecommend"].astype(float)

test_set["doRecommend"] = test_set["doRecommend"].astype(float)
train_set.head()
maxlen = 10 # Maximal length of sequence considered

num_words = 600 # Number of words in your vocabulary

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789' # Some chars you want to remove in order to have a clean text

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=num_words, filters=filters) # Word tokenizer

tokenizer.fit_on_texts(train_set['title'].tolist()) # Fit on training samples





train_titles_tok = tokenizer.texts_to_sequences(train_set['title'].tolist())

test_titles_tok = tokenizer.texts_to_sequences(test_set['title'].tolist())





from keras.preprocessing.sequence import pad_sequences

train_titles_pad = pad_sequences(train_titles_tok, maxlen=maxlen)

test_titles_pad = pad_sequences(test_titles_tok, maxlen=maxlen)
print(train_set.iloc[0]["title"],train_titles_pad[0])

print(train_set.iloc[1]["title"],train_titles_pad[1])

print(train_set.iloc[2]["title"],train_titles_pad[2])

print(train_set.iloc[3]["title"],train_titles_pad[3])
from sklearn.preprocessing import LabelBinarizer

label_enc = LabelBinarizer().fit(list(set(train_set['doRecommend'].values))) 

train_labels = label_enc.transform(train_set['doRecommend'].values)

test_labels = label_enc.transform(test_set['doRecommend'].values)
from keras.optimizers import Adam, RMSprop, SGD, Adagrad

from keras.callbacks import EarlyStopping



class LearningModel():

    def __init__(self, dim=20):

        self.dim= dim # Dimension of word embeddings

        self.n_label = 2 # Number of labels

        self.optimizer = Adam(lr=0.01) # Optimizer method for stochastic gradient descent

        self.epochs = 20

        self.batch_size = 128

        self.callbacks = EarlyStopping(monitor='val_loss', patience=2)

        self.model = None # Keras model, it will be instantiated later

        

    def compile(self):

        print(self.model.summary())

        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        

    def train(self, mode=''):

        self.compile()

        if mode == 'EarlyStopping':

            self.model.fit(train_titles_pad, train_labels,

                           epochs=self.epochs, 

                           batch_size=self.batch_size, 

                           validation_data=(test_titles_pad, test_labels), 

                           callbacks=[self.callbacks], verbose=2)

        else: 

            for _ in range(self.epochs):

                self.model.fit(train_titles_pad, train_labels,

                               epochs=1, 

                               batch_size=self.batch_size, 

                               validation_split=0.1, verbose=2)

                self.test()

    

    def test(self):

        print('Evaluation : ', self.model.evaluate(test_titles_pad, test_labels, batch_size=2048))
from keras.models import Sequential

from keras.layers import Embedding, GlobalAveragePooling1D, Dense, BatchNormalization



lm = LearningModel(dim=20)



lm.model = Sequential()



lm.model.add(Embedding(num_words, lm.dim, input_length=maxlen))

lm.model.add(GlobalAveragePooling1D())

lm.model.add(Dense(1, activation='sigmoid'))



lm.train()
print(train_set["doRecommend"].value_counts()/len(train_set))

print(test_set["doRecommend"].value_counts()/len(test_set))
df = df.reset_index()

df.head()
df1 = df[(df["doRecommend"] == False) | (df["index"]<3000)]

df1.info()
print(df1["doRecommend"].value_counts()/len(df1))

df1["doRecommend"].astype(float).hist(figsize=(8,5))
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df1, df1["doRecommend"]):

    train_set1 = df1.iloc[train_index]

    test_set1 = df1.iloc[test_index]
train_set1["title"] = train_set1["title"].astype(str)

test_set1["title"] = test_set1["title"].astype(str)



train_set1["doRecommend"] = train_set1["doRecommend"].astype(float)

test_set1["doRecommend"] = test_set1["doRecommend"].astype(float)
tokenizer = Tokenizer(num_words=num_words, filters=filters) # Word tokenizer

tokenizer.fit_on_texts(train_set1['title'].tolist()) # Fit on training samples





train_titles_tok = tokenizer.texts_to_sequences(train_set1['title'].tolist())

test_titles_tok = tokenizer.texts_to_sequences(test_set1['title'].tolist())





from keras.preprocessing.sequence import pad_sequences

train_titles_pad = pad_sequences(train_titles_tok, maxlen=maxlen)

test_titles_pad = pad_sequences(test_titles_tok, maxlen=maxlen)
label_enc = LabelBinarizer().fit(list(set(train_set1['doRecommend'].values))) 

train_labels = label_enc.transform(train_set1['doRecommend'].values)

test_labels = label_enc.transform(test_set1['doRecommend'].values)
lm = LearningModel(dim=20)



lm.model = Sequential()



lm.model.add(Embedding(num_words, lm.dim, input_length=maxlen))

lm.model.add(GlobalAveragePooling1D())

lm.model.add(Dense(1, activation='sigmoid'))



lm.train()