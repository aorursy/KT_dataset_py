# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))

test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
mispell_dict = {'colour':'color',

                'colours':'color',

                'colors':'color',

                'scores':'score',

                'centre':'center',

                'didnt':'did not',

                'doesnt':'does not',

                'isnt':'is not',

                'shouldnt':'should not',

                'favourite':'favorite',

                'travelling':'traveling',

                'counselling':'counseling',

                'theatre':'theater',

                'cancelled':'canceled',

                'labour':'labor',

                'organisation':'organization',

                'wwii':'world war 2',

                'citicise':'criticize',

                'instagram': 'social medium',

                'whatsapp': 'social medium',

                'Snapchat': 'social medium',

                'behaviour':'behavior',

                'realise':'realize',

                'favour':'favor',

                'learnt':'learned',

                'programme':'program',

                'recognise':'recognize'

                

                }
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",

                       "'cause": "because", "could've": "could have", "couldn't": "could not", 

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not",

                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 

                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                       "I'll've": "I will have","I'm": "I am", "I've": "I have", 

                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", 

                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",

                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",

                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                       "she's": "she is", "should've": "should have", "shouldn't": "should not",

                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", 

                       "that's": "that is", "there'd": "there would", "there'd've": "there would have",

                       "there's": "there is", "here's": "here is","they'd": "they would", 

                       "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", 

                       "they've": "they have", "to've": "to have", "wasn't": "was not",

                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", 

                       "we'll've": "we will have", "we're": "we are", "we've": "we have", 

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have",

                       "what're": "what are",  "what's": "what is", "what've": "what have", 

                       "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", 

                       "who'll've": "who will have", "who's": "who is", "who've": "who have", 

                       "why's": "why is", "why've": "why have", "will've": "will have", 

                       "won't": "will not", "won't've": "will not have", "would've": "would have",

                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",

                       "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 

                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def change_mispell(x):

    text=x.split()

    final_string = ' '.join(str(mispell_dict.get(word, word)) for word in text)

    return final_string



def change_mispell2(x):

    text=x.split()

    final_string = ' '.join(str(contraction_mapping.get(word, word)) for word in text)

    return final_string
train_df["question_text"] = [change_mispell(x) for x in train_df["question_text"]]

test_df["question_text"] = [change_mispell(x) for x in test_df["question_text"]]
train_df['question_text'] = [change_mispell2(x) for x in train_df['question_text']]

test_df['question_text'] = [change_mispell2(x) for x in test_df['question_text']]
## split to train and val

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
print(train_X.shape)

print(val_X.shape)

print(train_y.shape)

print(val_y.shape)
### preprocessing embedding

# - use google



embeddings_index_google = {}

with open('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', "rb") as f:

    header = f.readline()

    vocab_size, layer1_size = map(int, header.split())

    binary_len = np.dtype('float32').itemsize * layer1_size

    for line in range(vocab_size):

        word = []

        while True:

            ch = f.read(1).decode('latin-1')

            if ch == ' ':

                word = ''.join(word)

                break

            if ch != '\n':

                word.append(ch)

        vector = np.frombuffer(f.read(binary_len), dtype='float32')

        embeddings_index_google[word] = vector

        if line % 100000 == 0:

            print(word)



word_index = tokenizer.word_index

embedding_matrix = np.zeros((max_features, embed_size))

for word, i in word_index.items():

    embedding_vector = embeddings_index_google.get(word)

    if i < max_features:

        if embedding_vector is not None:

            # Words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector
## Define a model

from keras.models import Sequential

from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout



model = Sequential()

model.add(Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, weights=[embedding_matrix],name="embedding"))

model.add(Conv1D(64, 3, padding='same', name="conv1"))

model.add(Conv1D(32, 3, padding='same', name="conv2"))

model.add(Conv1D(16, 3, padding='same', name="conv3"))

model.add(Flatten(name="flatten"))

model.add(Dropout(0.5, name="dropout1"))

model.add(Dense(32, activation='relu', name="dense1"))

model.add(Dense(1, activation='sigmoid', name="output"))



model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
pred_test_y_google = model.predict([test_X], batch_size=1024, verbose=1)

del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
# use glove



EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
## Define a model

from keras.models import Sequential

from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout



model = Sequential()

model.add(Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, weights=[embedding_matrix],name="embedding"))

model.add(Conv1D(64, 3, padding='same', name="conv1"))

model.add(Conv1D(32, 3, padding='same', name="conv2"))

model.add(Conv1D(16, 3, padding='same', name="conv3"))

model.add(Flatten(name="flatten"))

model.add(Dropout(0.5, name="dropout1"))

model.add(Dense(32, activation='relu', name="dense1"))

model.add(Dense(1, activation='sigmoid', name="output"))



model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))



pred_test_y_glove = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
# wike



EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
## Define a model

from keras.models import Sequential

from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout



model = Sequential()

model.add(Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, weights=[embedding_matrix],name="embedding"))

model.add(Conv1D(64, 3, padding='same', name="conv1"))

model.add(Conv1D(32, 3, padding='same', name="conv2"))

model.add(Conv1D(16, 3, padding='same', name="conv3"))

model.add(Flatten(name="flatten"))

model.add(Dropout(0.5, name="dropout1"))

model.add(Dense(32, activation='relu', name="dense1"))

model.add(Dense(1, activation='sigmoid', name="output"))



model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))



pred_test_y_wiki = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
# use paragram



EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
## Define a model

from keras.models import Sequential

from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout



model = Sequential()

model.add(Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen, weights=[embedding_matrix],name="embedding"))

model.add(Conv1D(64, 3, padding='same', name="conv1"))

model.add(Conv1D(32, 3, padding='same', name="conv2"))

model.add(Conv1D(16, 3, padding='same', name="conv3"))

model.add(Flatten(name="flatten"))

model.add(Dropout(0.5, name="dropout1"))

model.add(Dense(32, activation='relu', name="dense1"))

model.add(Dense(1, activation='sigmoid', name="output"))



model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))



pred_test_y_paragram = model.predict([test_X], batch_size=1024, verbose=1)
# stacking

pred_test_y = 0.25*pred_test_y_google + 0.25*pred_test_y_glove + 0.25*pred_test_y_wiki + 0.25*pred_test_y_paragram

pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)