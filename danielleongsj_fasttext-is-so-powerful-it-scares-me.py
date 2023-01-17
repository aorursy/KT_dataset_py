import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import csv

from time import time

import json



import re

import string

from tqdm import tqdm



from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from gensim.models import FastText



from keras.layers import Input, Dense, Embedding, Flatten, Dropout, SpatialDropout1D # General

from keras.layers import CuDNNLSTM, Bidirectional # LSTM-RNN

from keras.optimizers import Adam



from keras import backend as K

from keras.callbacks import EarlyStopping



import tensorflow as tf



# Evaluation

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/ndsc-beginner/train.csv')
table = str.maketrans('','', string.punctuation)



def removeNumbersAndPunctuations(text):

    text = text.translate(table)

    text = re.sub(r'\d+', '', text)

    return text
df['title'] = df['title'].apply(removeNumbersAndPunctuations)
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['Category'], test_size=0.16, random_state=42)
print('loading word embeddings...')

embeddings_index = {}

f = open('../input/ftembeddings300all/ftembeddings300all.txt', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_index))
y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)
NUM_CATEGORIES = 58

MAX_SEQUENCE_LENGTH = 30

MAX_NB_WORDS = 20000

EMBED_DIM = 300

HIDDEN = 256
tok = Tokenizer(num_words=MAX_NB_WORDS, lower=True) 

tok.fit_on_texts(X_train)
word_index = tok.word_index

print('Found %s unique tokens.' % len(word_index))
sequences = tok.texts_to_sequences(X_train)

train_dtm = sequence.pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)



test_sequences = tok.texts_to_sequences(X_test)

test_dtm = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of Train DTM:', train_dtm.shape)
print('preparing embedding matrix...')

words_not_found = []

NUM_WORDS = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((NUM_WORDS, EMBED_DIM))

for word, i in word_index.items():

    if i >= NUM_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

    else:

        words_not_found.append(word)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("sample words not found: ", np.random.choice(words_not_found, 10))
def RNN_Model():

    text_sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TEXT_SEQUENCE_INPUT')

    

    rnn_layer = Embedding(NUM_WORDS, EMBED_DIM, weights=[embedding_matrix], trainable=False, name='EMBEDDING')(text_sequence) 

    rnn_layer = SpatialDropout1D(0.5, name='EMBEDDING_DROPOUT')(rnn_layer)

    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN, return_sequences=True), name='BILSTM_LAYER1')(rnn_layer)

    rnn_layer = Bidirectional(CuDNNLSTM(HIDDEN), name='BILSTM_LAYER2')(rnn_layer)

    rnn_layer = Dropout(0.5,name='RNN_DROPOUT')(rnn_layer)



    output = Dense(NUM_CATEGORIES, activation='softmax', name='OUTPUT')(rnn_layer)

    model = Model(inputs=text_sequence, outputs=output)

    

    return model
K.clear_session()

model = RNN_Model()

model.summary()
ea = EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)

adam = Adam(lr=0.001, decay=0.000049, epsilon=1e-8)
model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['categorical_accuracy'])

model.fit(train_dtm, y_train, batch_size=128, epochs=30, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea])
ea2 = EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)

adam2 = Adam(lr=0.001, decay=0.00006, epsilon=1e-8)
model.layers[1].trainable = True

model.compile(loss='categorical_crossentropy',optimizer=adam2, metrics=['categorical_accuracy'])

model.fit(train_dtm, y_train, batch_size=128, epochs=20, validation_data=(test_dtm,y_test), verbose=1, callbacks=[ea2])
model.evaluate(test_dtm, y_test)
y_pred = [np.argmax(pred) for pred in model.predict(test_dtm)]

y_truth = [np.argmax(truth) for truth in y_test.values]
with open('../input/ndsc-beginner/categories.json', 'rb') as handle:

    catNames = json.load(handle)



catNameMapper = {}

for category in catNames.keys():

    for key, value in catNames[category].items():

        catNameMapper[value] = key
catNameLabelsSorted = ['SPC', 'Icherry', 'Alcatel', 'Maxtron', 'Strawberry', 'Honor', 'Infinix', 'Realme', 

                       'Sharp', 'Smartfren', 'Motorola', 'Mito', 'Brandcode', 'Evercoss', 'Huawei', 

                       'Blackberry', 'Advan', 'Lenovo', 'Nokia', 'Sony', 'Asus', 'Vivo', 'Xiaomi', 'Oppo', 

                       'Iphone', 'Samsung', 'Others Mobile & Tablet', 'Big Size Top', 'Wedding Dress', 

                       'Others', 'Crop Top ', 'Big Size Dress', 'Tanktop', 'A Line Dress', 'Party Dress', 

                       'Bodycon Dress', 'Shirt', 'Maxi Dress', 'Blouse\xa0', 'Tshirt', 'Casual Dress', 

                       'Lip Liner', 'Setting Spray', 'Contour', 'Other Lip Cosmetics', 'Lip Gloss', 'Lip Tint', 

                       'Face Palette', 'Bronzer', 'Highlighter', 'Primer', 'Blush On', 'Concealer', 'Lipstick', 

                       'Foundation', 'Other Face Cosmetics', 'BB & CC Cream', 'Powder']
catNamePred = list(map(lambda x: catNameMapper[x], y_pred))

catNameActual = list(map(lambda x: catNameMapper[x], y_truth))
confMat = confusion_matrix(catNamePred, catNameActual, labels=catNameLabelsSorted)
fig, ax = plt.subplots(figsize=(30,30))

sns.heatmap(confMat, annot=True, fmt='d', xticklabels=catNameLabelsSorted, yticklabels=catNameLabelsSorted)

plt.ylabel('PREDICTED')

plt.xlabel('ACTUAL')

plt.show()
test_data = pd.read_csv('../input/ndsc-beginner/test.csv')

test_data['title'] = test_data['title'].apply(removeNumbersAndPunctuations)



test_sequences = tok.texts_to_sequences(test_data.title)

test_dtm = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)



y_pred = [np.argmax(pred) for pred in model.predict(test_dtm)]

test_data['Category'] = y_pred
test_data
df_submit = test_data[['itemid', 'Category']].copy()

df_submit.to_csv('submission_svc.csv', index=False)