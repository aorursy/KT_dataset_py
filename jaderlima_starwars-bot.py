import keras

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.layers.core import Activation, Dense, Dropout

from keras.layers.wrappers import TimeDistributed

from keras.layers.core import Dense, Activation

import keras.utils as kutils

import random



import pandas as pd

import numpy as np

import string, os 



import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
old_starwars = []

#article_df = pd.read_csv(curr_dir + filename)

#        all_headlines.extend(list(article_df.headline.values))



df_iv = pd.read_csv('../input/SW_EpisodeIV.txt', sep=" ",names=["Character","Dialogue"],header='infer',skiprows=1 )

df_iv['Movie'] = 'iv'

df_v = pd.read_csv('../input/SW_EpisodeV.txt', sep=" ",names=["Character","Dialogue"],header='infer',skiprows=1 )

df_v['Movie'] = 'v'

df_vi = pd.read_csv('../input/SW_EpisodeVI.txt', sep=" ",names=["Character","Dialogue"],header='infer',skiprows=1 )

df_vi['Movie'] = 'vi'

#old_starwars.extend(list)

df_starwars_all= pd.concat([df_iv, df_v,df_vi ])

old_starwars.extend(list(df_starwars_all.Character.values +": "+ df_starwars_all.Dialogue.values))

#old_starwars.extend(list(df_starwars_all.Dialogue.values))

def lower_text(txt):

    txt = "".join(b for b in txt if b not in string.punctuation).lower()

    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt 



old_starwars_corpus = [lower_text(line) for line in old_starwars]
from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

import string



all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation] \

             for sen in old_starwars_corpus]



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

model.save('stars-nlg.h5')



with open('stars-nlg-dict.pkl', 'wb') as handle:

    pickle.dump(word_indexes, handle)



with open('stars-nlg-maxlen.pkl', 'wb') as handle:

    pickle.dump(maxlen, handle)

print("Model Saved!")
import pickle



model = keras.models.load_model('stars-nlg.h5')

maxlen = pickle.load(open('stars-nlg-maxlen.pkl', 'rb'))

word_indexes = pickle.load(open('stars-nlg-dict.pkl', 'rb'))


#sample_seed = "did you hear that they ve shut down the main reactor we ll be destroyed for sure this is madness"

sample_seed = "your father are you better imagine"

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
def get_word_by_index(index, word_indexes):

    for w, i in word_indexes.items():

        if index == i:

            return w
def chit_chat(sample_seed):

    i = 0

    chat = ""

    while i < 100: 

        #sample_seed = "your father are you better imagine"

        sample_seed_vect = np.array([[word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK'] \

                            for c in word_tokenize(sample_seed)]])



        #print(sample_seed_vect)



        sample_seed_vect = pad_sequences(sample_seed_vect, maxlen=maxlen)



        #print(sample_seed_vect)



        predicted = model.predict_classes(sample_seed_vect, verbose=0)

        

        for p in predicted:  

            chat = (get_word_by_index(p, word_indexes))

            

        i = i + 1

        sample_seed = sample_seed + " " + chat



    return sample_seed
n = random.randint(1,len(old_starwars))



print(chit_chat(old_starwars[n-1:n][0]))
