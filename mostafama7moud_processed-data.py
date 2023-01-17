import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import progressbar

from keras.preprocessing import sequence

from keras.utils import to_categorical

import json
new_dataset = pd.read_csv("../input/down-upsampling/data400k-v2.csv")
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
df = new_dataset



print(df.head())

new_review = []

new_val = ''

for review in df['Reviews']:

    new_val = review.lower()

    new_val = re.sub('[^a-z0-9 ]+', '', new_val)

    new_val = re.sub(' \d+', ' ', new_val)

    new_review.append(new_val)



df['Reviews'] = pd.Series(new_review)

print(df.head())
X = df['Reviews']

Y = df['Rating']



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



bar2 = progressbar.ProgressBar(maxval = Y.shape[0])

count = 0

new_y = []

bar2.start()

for val in Y:

    new_val = val-1

    new_y.append(new_val)

    count += 1

    bar2.update(count)

bar2.finish()    



X = pd.Series(new_x)

Y = pd.Series(new_y)
word2idx_bar = progressbar.ProgressBar(maxval=X.shape[0])

word_to_idx = {'pad':0}

num_word = 1

count = 0

max_len = 0

avg = 0

word2idx_bar.start()

new_x = []



for tokens in X:

    length =len(tokens)

    avg += length

    if length > max_len:

        max_len = length

    new_tokens = []

    for token in tokens:

        if token not in word_to_idx:

            word_to_idx[token] = num_word

            num_word += 1

        new_tokens.append(word_to_idx[token])

    new_x.append(new_tokens)

    count += 1

    word2idx_bar.update(count)

    

word2idx_bar.finish()

with open('word-to-index-400k.json', 'w') as outfile:  

    json.dump(word_to_idx, outfile)



print(X[1], len(X[1]))

X = pd.Series(new_x)

print(new_x[1], len(new_x[1]))

print("===============================================")

avg = avg / X.shape[0]

print("Vocab Size : %d" % num_word)

print("Avg Num of Token : %d" % avg)

print("Max Num of Token : %d" % max_len)
x = X.to_numpy()

y = Y.to_numpy()



print("Saving Dataset .....")

np.save("processed-400k-x.npy", x)

np.save("processed-400k-y.npy", y)

print("Done :)")