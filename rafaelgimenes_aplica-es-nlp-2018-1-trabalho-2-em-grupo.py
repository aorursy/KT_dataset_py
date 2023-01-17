#imports necessarios

import numpy as np

import pandas as pd

import string, os 

import keras

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.layers.core import Activation, Dense, Dropout

from keras.layers.wrappers import TimeDistributed

from keras.layers.core import Dense, Activation

import keras.utils as kutils

import pickle

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

import string

#loading datasets

df_ep_VI = pd.read_table("../input/SW_EpisodeVI.txt",header=0, escapechar='\\',delim_whitespace=True,)

df_ep_IV = pd.read_table('../input/SW_EpisodeIV.txt',header=0, escapechar='\\',delim_whitespace=True)

df_ep_V = pd.read_table("../input/SW_EpisodeV.txt",header=0, escapechar='\\',delim_whitespace=True)
#checking if all have same structure

df_ep_VI.columns
df_ep_V.columns
df_ep_IV.columns
#checking sizes

print(df_ep_VI.shape)

print(df_ep_V.shape)

print(df_ep_IV.shape)

#join the datasets, to work with only one.

df = pd.concat([df_ep_IV, df_ep_V, df_ep_VI])
#checking size

df.shape
#getting the lines 

all_head_lines = []

for filename in df:

    all_head_lines.extend(list(df.dialogue.values))

    break

len(all_head_lines)
all_head_lines[:5]
#creating the corpus

def lower_text(txt):

    txt = "".join(b for b in txt if b not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 



corpus = [lower_text(line) for line in all_head_lines]

len(corpus)
corpus[:5]
#creating sents

sents = [[wd.lower() for wd in word_tokenize(sent) if not wd in string.punctuation]  for sent in all_head_lines]

x = []

y = []

print(sents[:10])
for sent in sents:

    for i in range(1, len(sent)):

        x.append(sent[:i])

        y.append(sent[i])
print(x[:5])

print(y[:5])

#train,test split

text = [i for sent in x for i in sent]

text += [i for i in y]

#unknow

text.append('UNK') 

words = list(set(text))      

word_indexes = {word: index for index, word in enumerate(words)}      

max_features = len(word_indexes)



x = [[word_indexes[i] for i in sent] for sent in x]

y = [word_indexes[i] for i in y]
print(x[:5])

print(y[:5])
y = kutils.to_categorical(y, num_classes=max_features)

maxlen = max([len(sent) for sent in x])

print(maxlen)
x = pad_sequences(x, maxlen=maxlen)

x = pad_sequences(x, maxlen=maxlen)



for ya in y:

    for i in range(len(ya)):

        if ya[i] != 0:

            print(i)
embedding_size = 10

model = Sequential()  

# Add Input Embedding Layer

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

# Add Hidden Layer 1 - LSTM Layer

model.add(LSTM(100))

model.add(Dropout(0.1))

    

# Add Output Layer

model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.fit(x, y, epochs=30, verbose=5)
print("Saving model...")

model.save('shak-nlg.h5')



with open('shak-nlg-dict.pkl', 'wb') as handle:

    pickle.dump(word_indexes, handle)



with open('shak-nlg-maxlen.pkl', 'wb') as handle:

    pickle.dump(maxlen, handle)

print("Model Saved!")
model = keras.models.load_model('shak-nlg.h5')

maxlen = pickle.load(open('shak-nlg-maxlen.pkl', 'rb'))

word_indexes = pickle.load(open('shak-nlg-dict.pkl', 'rb'))
#igual da aula

sample_seed = input()

sample_seed_vect = np.array([[word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK'] for c in word_tokenize(sample_seed)]])



print(sample_seed_vect)

sample_seed_vect = pad_sequences(sample_seed_vect, maxlen=maxlen)

print(sample_seed_vect)



predicted = model.predict_classes(sample_seed_vect, verbose=0)

print(predicted)



def get_word_by_index(index, word_indexes):

    for w, i in word_indexes.items():

        if index == i:

            return w

        

    return None



for p in predicted:    

    print(get_word_by_index(p, word_indexes))
sample_seed = input()

sample_seed_vect = [word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK']  for c in word_tokenize(sample_seed)]



print(sample_seed_vect)

predicted = []



while len(sample_seed_vect) < 100:

    predicted = model.predict_classes(pad_sequences([sample_seed_vect], maxlen=maxlen, padding='pre'), verbose=0)

    sample_seed_vect.extend(predicted)



res = []



for p in sample_seed_vect:    

   res.append(get_word_by_index(p, word_indexes)) 



print(' '.join (res))