import re

import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import pandas as pd

import numpy as np

import progressbar

import json

from keras.models import load_model

from keras.preprocessing import sequence
model = load_model('/kaggle/input/lstm-cnn-5-level-sentiment/LSTM-5-level-sentiment-400k-v1.h5')
def get_wordnet_pos(treebank_tag):



    if treebank_tag.startswith('J'):

        return wordnet.ADJ

    elif treebank_tag.startswith('V'):

        return wordnet.VERB

    elif treebank_tag.startswith('N'):

        return wordnet.NOUN

    elif treebank_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
df = np.array(["I am was going to get two pound",

           "I have bought samsung s20 i think it's overrated",

           "I like this phone but really it's software is piece of shit",

           "It's ok",

           "I don't recommend any one to buy this phone its sucks really",

           "good phone I am excited to have one of it"])



new_review = []

new_val = ''

for review in df:

    new_val = review.lower()

    new_val = re.sub('[^a-z0-9 ]+', '', new_val)

    new_val = re.sub(' \d+', ' ', new_val)

    new_review.append(new_val)



df = pd.Series(new_review)

print(df.head())
X = df

bar1 = progressbar.ProgressBar(maxval = X.shape[0])

new_x = []

count = 0

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()

bar1.start()

for review in X:

    tokens = nltk.word_tokenize(review) 

    tokens = [w for w in tokens if not w in stop_words]

    tags = nltk.pos_tag(tokens)

    new_tags = []

    for i in range(len(tags)):

        tag = wordnet_lemmatizer.lemmatize(tags[i][0], get_wordnet_pos(tags[i][1]))

        new_tags.append(tag)

    new_x.append(new_tags)

    

    count += 1

    bar1.update(count)

    

bar1.finish()



X = pd.Series(new_x)

print(X)
word_to_idx = {}

with open('../input/processed-data/word-to-index-400k.json', 'r') as f:

    word_to_idx = json.load(f)



new_x = []

for tokens in X:

    new_tokens = []

    for token in tokens:

        if token not in word_to_idx:

            new_tokens.append(0)

        else:

            new_tokens.append(word_to_idx[token])

    new_x.append(new_tokens)

    

X = pd.Series(new_x)

print(X)
X = sequence.pad_sequences(X, maxlen=150)

prediction = model.predict(X)

prediction_classes = np.argmax(prediction,axis = 1) 

for i in range(len(df)):

    print("Prediction for [ %s ] ===> [%d]" % (df[i], prediction_classes[i]))