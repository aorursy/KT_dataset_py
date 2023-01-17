# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints



from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed

from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D

from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate

from keras.layers import Reshape, merge, Concatenate, Lambda, Average

from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint

from keras.initializers import Constant

from keras.layers.merge import add



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.utils import np_utils



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)

train_df.head()
# using headlines and short_description as input X

train_df['text'] = train_df.headline + " " + train_df.short_description

train_df.text = train_df.text.map(lambda x: x.lower())

train_df.head()
train_df.category.isnull().values.any()
train_df.text.isnull().values.any()
categories = train_df.groupby('category')

print('total categories: {}'.format(categories.ngroups))

print(categories.size())
# Combine similar categories

train_df.category = train_df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

train_df.category = train_df.category.map(lambda x: "WORLD NEWS & POLITICS" if x == "WORLD NEWS" or x == "POLITICS" else x)

train_df.category = train_df.category.map(lambda x: "RACIAL & LGBTQ ISSUES" if x == "BLACK VOICES" or x == "LATINO VOICES" or x == "QUEER VOICES" else x)

train_df.category = train_df.category.map(lambda x: "GREEN & HEALTHY LIVING" if x == "GREEN" or x == "HEALTHY LIVING" else x)

train_df.category = train_df.category.map(lambda x: "STYLE & BEAUTY" if x == "STYLE" else x)

train_df.category = train_df.category.map(lambda x: "TRAVEL & CULINARY" if x == "TRAVEL" or x == "TASTE" or x == "FOOD & DRINK" else x)

train_df.category = train_df.category.map(lambda x: "COLLEGE & EDUCATION" if x == "EDUCATION" or x == "COLLEGE" else x)

train_df.category = train_df.category.map(lambda x: "ARTS & CULTURE" if x == "ARTS" or x == "CULTURE & ARTS" or x == "ARTS & CULTURE" else x)

train_df.category = train_df.category.map(lambda x: "PARENTING" if x == "PARENTING" or x == "PARENT" or x == "PARENTS" else x)

train_df.category = train_df.category.map(lambda x: "WEDDING & DIVORCE" if x == "WEDDING" or x == "DIVORCE" or x == "WEDDINGS" else x)

train_df.category = train_df.category.map(lambda x: "SCIENCE & TECHNOLOGY" if x == "SCIENCE" or x == "TECH" else x)

train_df.category = train_df.category.map(lambda x: "SPORT" if x == "SPORTS" else x)



categories = train_df.groupby('category')

print('total categories: {}'.format(categories.ngroups))

print(categories.size())
# tokenizing

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_df.text)

X = tokenizer.texts_to_sequences(train_df.text)

train_df['words'] = X



# delete some empty and short data

train_df['word_length'] = train_df.words.apply(lambda i: len(i))

train_df = train_df[train_df.word_length >= 5]



train_df.head()
train_df.word_length.describe()
# using 50 for padding length

maxlen = 50

X = list(sequence.pad_sequences(train_df.words, maxlen=maxlen))
# category to id

categories = train_df.groupby('category').size().index.tolist()

category_int = {}

int_category = {}

for i, k in enumerate(categories):

    category_int.update({k:i})

    int_category.update({i:k})



train_df['c2id'] = train_df['category'].apply(lambda x: category_int[x])



train_df.head()
word_index = tokenizer.word_index



EMBEDDING_DIM = 100



embeddings_index = {}

f = open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s unique tokens.' % len(word_index))

print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



embedding_layer = Embedding(len(word_index)+1,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=maxlen,

                            trainable=False)
# prepared data 

X = np.array(X)

Y = np_utils.to_categorical(list(train_df.c2id))



# and split to training set and validation set

seed = 29

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
inp = Input(shape=(maxlen,), dtype='int32')

x = embedding_layer(inp)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

x = Conv1D(64, kernel_size=3)(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

x = concatenate([avg_pool, max_pool])

outp = Dense(len(int_category), activation="softmax")(x)



BiGRU = Model(inp, outp)

BiGRU.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



BiGRU.summary()
# training

bigru_history = BiGRU.fit(x_train, 

                          y_train, 

                          batch_size=128, 

                          epochs=20, 

                          validation_data=(x_val, y_val))
def evaluate_accuracy(model):

    predicted = model.predict(x_val)

    diff = y_val.argmax(axis=-1) - predicted.argmax(axis=-1)

    corrects = np.where(diff == 0)[0].shape[0]

    total = y_val.shape[0]

    return float(corrects/total)
print('Bidirectional GRU: {}'.format(evaluate_accuracy(BiGRU)))
!rm -rf bigru*
from IPython.display import FileLink, FileLinks

# serialize model to JSON

model_json = BiGRU.to_json()

with open("bigru.json", "w") as json_file:             

     json_file.write(model_json) 



# serialize weights to HDF5

BiGRU.save_weights("bigru.h5")

print("Saved model to disk")

FileLinks('.') #lists all downloadable files on server
raw_dfs = []



for dirname, _, filenames in os.walk('/kaggle/input/all-the-news'):

    for filename in filenames:

        raw_dfs.append(pd.read_csv(os.path.join(dirname, filename), index_col=[0]))



df = pd.concat(raw_dfs, axis=0, ignore_index=True)



df.title = df.title.astype(str)

df.content = df.content.astype(str)



df.head()
# using title and content as input X

df['text'] = df.title + " " + df.content

# df['text'] = df.title

df.text = df.text.map(lambda x: x.lower())



# tokenizing

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df.text)

X_to_predict = tokenizer.texts_to_sequences(df.text)

df['words'] = X_to_predict



# delete some empty and short data

df['word_length'] = df.words.apply(lambda i: len(i))

df = df[df.word_length >= 5]



df.head()
# using 50 for padding length

maxlen = 50

X_to_predict = list(sequence.pad_sequences(df.words, maxlen=maxlen))

df['padded_words'] = X_to_predict



df.head()
sample_text = df.iloc[999].text

sample_title = df.iloc[999].title

print('sample title: {}'.format(sample_title))



tokenizer.fit_on_texts(sample_text)

X_sample = tokenizer.texts_to_sequences(sample_text)

X_max_words = list(sequence.pad_sequences(X_sample, maxlen=maxlen))



predict = BiGRU.predict([X_max_words])



predict
sorting = (-predict).argsort()

value = sorting[0][0]

value1 = sorting[0][1]



predicted_label = int_category[value]

predicted_label1 = int_category[value1]

print(predicted_label)

print(predicted_label1)
def get_first_topic(sample_text):

    tokenizer.fit_on_texts(sample_text)

    X_sample = tokenizer.texts_to_sequences(sample_text)

    X_max_words = list(sequence.pad_sequences(X_sample, maxlen=maxlen))



    predict = BiGRU.predict([X_max_words])

    

    sorting = (-predict).argsort()

    value = sorting[0][0]



    predicted_label = int_category[value]

    return predicted_label



def get_second_topic(padded_words):

    tokenizer.fit_on_texts(sample_text)

    X_sample = tokenizer.texts_to_sequences(sample_text)

    X_max_words = list(sequence.pad_sequences(X_sample, maxlen=maxlen))



    predict = BiGRU.predict([X_max_words])

    

    sorting = (-predict).argsort()

    value = sorting[0][1]



    predicted_label = int_category[value]

    return predicted_label