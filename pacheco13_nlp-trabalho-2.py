import numpy as np

import pandas as pd

import string, os 
#Lendo os 3 datasets

df_IV = pd.read_table('../input/SW_EpisodeIV.txt',delim_whitespace=True, header=0, escapechar='\\')

df_V = pd.read_table("../input/SW_EpisodeV.txt",delim_whitespace=True, header=0, escapechar='\\')

df_VI = pd.read_table("../input/SW_EpisodeVI.txt",delim_whitespace=True, header=0, escapechar='\\')
#Verificando o tamanho de cada dataset.

print(df_IV.shape)

print(df_V.shape)

print(df_VI.shape)
#Concatenando os datasets para poder utilizar apenas 1.

df = pd.concat([df_IV, df_V, df_VI])
#Verificando o tamanho do novo dataset.

df.shape
all_headlines = []

for filename in df:

    all_headlines.extend(list(df.dialogue.values))

    break

len(all_headlines)
all_headlines
import keras

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.layers.core import Activation, Dense, Dropout

from keras.layers.wrappers import TimeDistributed

from keras.layers.core import Dense, Activation

import keras.utils as kutils
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 



corpus = [clean_text(x) for x in all_headlines]

corpus[:10]
from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

import string



all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation] \

             for sen in all_headlines]



x = []

y = []



print(all_sents[:10])



for sen in all_sents:

    for i in range(1, len(sen)):

        x.append(sen[:i])

        y.append(sen[i])

        



print(x[:10])

print(y[:10])
from sklearn.model_selection import train_test_split

import numpy as np



all_text = [c for sen in x for c in sen]

all_text += [c for c in y]



all_text.append('UNK') # Palavra desconhecida



words = list(set(all_text))

        

word_indexes = {word: index for index, word in enumerate(words)}      



max_features = len(word_indexes)



x = [[word_indexes[c] for c in sen] for sen in x]

y = [word_indexes[c] for c in y]



print(x[:10])

print(y[:10])



y = kutils.to_categorical(y, num_classes=max_features)



maxlen = max([len(sen) for sen in x])



print(maxlen)
x = pad_sequences(x, maxlen=maxlen)

x = pad_sequences(x, maxlen=maxlen)



print(x[:10,-10:])

print(y[:10,-10:])
print(x[:10,-10:])



for y_ in y:

    for i in range(len(y_)):

        if y_[i] != 0:

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
model.fit(x, y, epochs=20, verbose=5)
import pickle



print("Saving model...")

model.save('shak-nlg.h5')



with open('shak-nlg-dict.pkl', 'wb') as handle:

    pickle.dump(word_indexes, handle)



with open('shak-nlg-maxlen.pkl', 'wb') as handle:

    pickle.dump(maxlen, handle)

print("Model Saved!")
import pickle



model = keras.models.load_model('shak-nlg.h5')

maxlen = pickle.load(open('shak-nlg-maxlen.pkl', 'rb'))

word_indexes = pickle.load(open('shak-nlg-dict.pkl', 'rb'))
sample_seed = input()

sample_seed_vect = np.array([[word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK'] \

                    for c in word_tokenize(sample_seed)]])



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

sample_seed_vect = [word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK'] \

                    for c in word_tokenize(sample_seed)]



print(sample_seed_vect)

predicted = []



while len(sample_seed_vect) < 100:

    

    predicted = model.predict_classes(pad_sequences([sample_seed_vect], maxlen=maxlen, padding='pre'), verbose=0)

    sample_seed_vect.extend(predicted)



    

res = []

   



for p in sample_seed_vect:    

   res.append(get_word_by_index(p, word_indexes)) 



print(' '.join (res))