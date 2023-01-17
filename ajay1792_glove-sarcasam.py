# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==2.0.0-alpha0
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import os

os.chdir(r'/kaggle/working')

from IPython.display import FileLink

FileLink('/kaggle/working/tokenize.pickle')


import pandas as pd

df = pd.read_json("/kaggle/input/news-headline-gloves/Sarcasm_Headlines_Dataset.json", lines=True)



test_filename = "/kaggle/input/test-4/test_4.txt"



test_data_raw = pd.read_csv(test_filename, delimiter="\t").fillna('')



test_data_raw.head()

df.head()
import tensorflow as tf

print(tf.__version__)
df = df.drop(['article_link'], axis=1)

df.head()

df1 = test_data_raw.drop(['SentenceId'], axis = 1)

df1=df1.replace('\w+ ','',regex=True)

df1.head()

df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))

#df.head()

df1['len'] = df1['Phrase'].apply(lambda x: len(x.split(" ")))

#df1.head()
import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential



max_features = 10000

maxlen = 25

embedding_size = 200



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(df['headline']))





X = tokenizer.texts_to_sequences(df['headline'])

X = pad_sequences(X, maxlen = maxlen, padding='post')

test_X = tokenizer.texts_to_sequences(df1['Phrase'])

test_X = pad_sequences(test_X, maxlen = maxlen, padding='post')

y = df['is_sarcastic']





import pickle

tokenize = open("/kaggle/input/tokenize.pickle", "wb")

pickle.dump(tokenizer, tokenize)

tokenize.close()
ls /kaggle/working/featuresets.pickle
EMBEDDING_FILE = '/kaggle/input/glove6b200d/glove.6B.200d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(40, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 5

save_model = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
import pickle

save_featuresets = open("/kaggle/working/featuresets.pickle", "wb")

pickle.dump(model, save_featuresets)

save_featuresets.close()
loss, acc = model.evaluate(X, y, verbose=2)

print("Overall scores")

print("Loss\t\t: ", round(loss, 3))

print("Accuracy\t: ", round(acc, 3))
y_pred1=model.predict_classes(test_X, verbose=1)

test_data_raw['val']=y_pred1

#sub.to_csv('sarcasam_glove.csv',index=False)

#y_pred2

test_data_raw
prob = model.predict_proba(test_X, verbose=1)

test_data_raw['prob']=prob

test_data_raw
text =  ['its a good day ','I work 40 hours a week for us to be this poor']

def tokenizer_pickle(text):

    #give text 

    tokenizer_path= '/kaggle/working/tokenize.pickle'

    model_path = '/kaggle/working/featuresets.pickle'

    tokenize = open(tokenizer_path, "rb")

    model_path = open(model_path, "rb")

    tokenizer = pickle.load(tokenize)

    model = pickle.load(model_path)

    test_X = tokenizer.texts_to_sequences(text)

    test_X = pad_sequences(test_X, maxlen = maxlen)

    

    pred = model.predict_classes(test_X,verbose= 1)

    print(pred)

    return pred
tokenizer_pickle(['Keep talking ,you\'ll say something intelligent.'])
import pandas as pd

import pickle

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential





df = pd.read_json("/kaggle/input/news-headline-gloves/Sarcasm_Headlines_Dataset.json", lines=True)

test_filename = "/kaggle/input/test-4/test_4.txt"

test_data_raw = pd.read_csv(test_filename, delimiter="\t").fillna('')



df1 = test_data_raw.drop(['SentenceId'], axis = 1)

df1=df1.replace('\w+ ','',regex=True)



df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))

df1['len'] = df1['Phrase'].apply(lambda x: len(x.split(" ")))





max_features = 10000

maxlen = 25

embedding_size = 200



#Vocabulary-Indexing of the train and test sentence(phrase), make sure "filters" parm doesn't clean out punctuations

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(df['headline']))



X = tokenizer.texts_to_sequences(df['headline'])

X = pad_sequences(X, maxlen = maxlen)

test_X = tokenizer.texts_to_sequences(df1['Phrase'])

test_X = pad_sequences(test_X, maxlen = maxlen)

y = df['is_sarcastic']



tokenize = open("/kaggle/input/tokenize.pickle", "wb")

pickle.dump(tokenizer, tokenize)

tokenize.close()



#word embeddings Glove

EMBEDDING_FILE = '/kaggle/input/glove6b200d/glove.6B.200d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector



#Since we are dealing with sequences (the tokens) the input should be an embedding layer. 

model = Sequential()

model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))



#Bidirectional LSTMs train two instead of one LSTMs on the input sequence.

#Bidirectional LSTMs provide additional context to the network and result in faster and even fuller learning on the problem

#CuDNN implements kernels for large matrix operations on GPU using CUDA(CuDNNLSTM for 0arallel processing).

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(40, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 7

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.predict_classes(test_X, verbose=1)

save_featuresets = open("/kaggle/input/featuresets.pickle", "wb")

pickle.dump(model, save_featuresets)

save_featuresets.close()
