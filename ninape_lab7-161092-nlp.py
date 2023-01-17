!pip list | grep tensorflow
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

import string

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



from keras.models import Sequential, Model

from keras.layers.embeddings import Embedding

from keras.layers import Dense, Input, LSTM, GRU, Bidirectional, Conv1D, Dropout, MaxPooling1D, Flatten, TimeDistributed

from sklearn.model_selection import train_test_split



from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.sequence import pad_sequences

from keras.layers import concatenate





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

# Any results you write to the current directory are saved as output.
!bert-serving-start -model_dir /kaggle/input/bert-nina/uncased_L-12_H-768_A-12/ -num_worker=4 
!pip install bert-serving-server  # server

!pip install bert-serving-client
def read_corpus():

    sentence_1 = pd.read_table('/kaggle/input/multinli/multinli_dev_set.txt')['sentence1'].get_values() 

    sentence_2 = pd.read_table('/kaggle/input/multinli/multinli_dev_set.txt')['sentence2'].get_values() 

    labels = pd.read_table('/kaggle/input/multinli/multinli_dev_set.txt')['gold_label'].get_values() 

    return sentence_1, sentence_2, labels
s1,s2,labels = read_corpus()

s1_l = s1

s2_l = s2

print(s1)

print(s2)
sentences1 = []

sentences2 = []

vocabulary = []

stops = set(stopwords.words('english'))

puncs = (string.punctuation)



for sentence in s1:

    words = []

    #print(sentence.split(" "))

    for word in sentence.split(" "):

       # print(word)

        words_i = word_tokenize(word)

        for word in words_i:

            if word not in stops and word not in puncs and not word.isdigit():

                words.append(word)

    sentences1.append(words)

    vocabulary.extend(words)

  

    

print('Sentences1:', len(sentences1))     



for sentence in s2:

    words = []

    #print(sentence.split(" "))

    for word in sentence.split(" "):

       # print(word)

        words_i = word_tokenize(word)

        for word in words_i:

            if word not in stops and word not in puncs and not word.isdigit():

                words.append(word)

    sentences2.append(words)

    vocabulary.extend(words)

    

print('Sentences2:', len(sentences2))       
print(len(s1))
max_s1 = 0

max_s2 = 0

z = 0

for sentence in sentences1:

    z += len(sentence)

   # print(len(sentence))

    if len(sentence) > max_s1:

        max_s1 = len(sentence)

        print(max_s1)

        print (sentence)

        

        

for sentence in sentences2:

    if len(sentence) > max_s2:

        max_s2 = len(sentence)



  

print(z/len(sentences1))

print(max_s1)

print(max_s2)
len(set(vocabulary))
vocabulary_index = dict([(y, x+1) for x, y in enumerate(sorted(set(vocabulary)))])

vocabulary_index
mapped_s1 = []

mapped_s2 = []



for s in sentences1:

    words = []

    for w in s:

        words.append(vocabulary_index[w])

    mapped_s1.append(words)

for s in sentences2:

    words = []

    for w in s:

        words.append(vocabulary_index[w])

    mapped_s2.append(words)  
values = array(labels)

print(set(values))

# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)

print(integer_encoded)

# binary encode

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
sen1 = pad_sequences(mapped_s1, 15,padding='post')

sen2 = pad_sequences(mapped_s2, 15,padding='post')
sen1
l = onehot_encoded

x= []

for (i,s_1) in enumerate(sen1):

    x.append([s_1,sen2[i]])

np.array(x).shape
# np.array(y).shape
sen1.shape
l.shape
X_train_s1, X_test_s1, y_train, y_test = train_test_split(sen1, l, test_size=0.2, random_state=1)

X_test_s1, X_validate_s1, y_test, y_validate = train_test_split(X_test_s1, y_test, test_size=0.5, random_state=1)
X_train_s2, X_test_s2, y_train, y_test = train_test_split(sen2, l, test_size=0.2, random_state=1)

X_test_s2, X_validate_s2, y_test, y_validate = train_test_split(X_test_s2, y_test, test_size=0.5, random_state=1)
l = len(set(vocabulary))
# model = Sequential()

# model.add(Embedding(l, 4, input_shape=np.array(x).shape))

# #model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.7))

# model.add(Dense(np.array(y).shape[1], activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# print(model.summary())
np.array(sen1).shape

ll = len(set(vocabulary))
inputA = Input(shape=(15,))

inputB = Input(shape=(15,))

# the first branch operates on the first input



#model.add(Embedding(l, 4, input_shape=np.array(x).shape))

#model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.7))

#model.add(Dense(np.array(y).shape[1], activation='softmax'))



x = Embedding(ll+1, 100)(inputA)

x = LSTM(128,return_sequences=True, dropout=0.7)(x)

x = Dense(4, activation="relu")(x)

x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input

y = Embedding(ll+1, 100)(inputB)

y = LSTM(128,return_sequences=True, dropout=0.7)(y)

y = Dense(4, activation="relu")(y)

y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches

combined = concatenate([x.output, y.output])

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(4, activation="relu")(combined)

z = Flatten()(z)

z = Dense(4, activation="linear")(z)

# our model will accept the inputs of the two branches and

# then output a single value

model = Model(inputs=[x.input, y.input], outputs=z)



print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer='adam')

# train the model

print("[INFO] training model...")

model.fit(

    [X_train_s1, X_train_s2], y_train,

    validation_data=([X_validate_s1, X_validate_s2], y_validate),

    epochs=10, batch_size=150)

# make predictions on the testing data

print("[INFO] predicting house prices...")

preds = model.predict([X_test_s1, X_test_s2])
import copy
preds

for p in preds:

    print(p)

    print(np.argmax(p))

    break



class_labels = np.argmax(preds, axis=1)

class_labels



proverka = np.argmax(y_test,axis = 1)

proverka
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
print('accuracy_score = ', end="")

print(accuracy_score(proverka, class_labels))

print('Precision_score = ', end="")

print(precision_score(proverka, class_labels, average='macro'))

print('Recall_score = ', end="")

print(recall_score(proverka, class_labels, average='macro'))

print('F1_score = ', end="")

print(f1_score(proverka, class_labels, average='macro'))
print(sen1)
from keras.preprocessing.text import Tokenizer

word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(s1_l)

vocab_length = len(word_tokenizer.word_index) + 1
word_tokenizer_2 = Tokenizer()

word_tokenizer_2.fit_on_texts(s1_l)

vocab_length_2 = len(word_tokenizer_2.word_index) + 1
def loadGloveModel(gloveFile):

    # ovaa funkcija ja koristam za da gi load embeddings

    print("Loading Glove Model")

    f = open(gloveFile, 'r', encoding="utf8")

    model = {}

    for line in f:

        splitLine = line.split()

        word = splitLine[0]

        embedding = np.array([float(val) for val in splitLine[1:]])

        model[word] = embedding

    print("Done.", len(model), " words loaded!")

    return model
glove = loadGloveModel('/kaggle/input/lab4-nlp/glove.6B.50d.txt')
from nltk.tokenize import word_tokenize
#sega model so glove

embedding_matrix = np.zeros((ll+1, 50))

for word, index in word_tokenizer.word_index.items():

    if word in glove:

        embedding_vector = glove[word]

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
#sega model so glove

embedding_matrix_2 = np.zeros((ll+1, 50))

for word, index in word_tokenizer_2.word_index.items():

    if word in glove:

        embedding_vector = glove[word]

    if embedding_vector is not None:

        embedding_matrix_2[index] = embedding_vector
inputA = Input(shape=(15,))

inputB = Input(shape=(15,))

# the first branch operates on the first input



#model.add(Embedding(l, 4, input_shape=np.array(x).shape))

#model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.7))

#model.add(Dense(np.array(y).shape[1], activation='softmax'))



x = Embedding(ll+1, 50,weights=[embedding_matrix])(inputA)

x = LSTM(128,return_sequences=True, dropout=0.7)(x)

x = Dense(4, activation="relu")(x)

x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input

y = Embedding(ll+1, 50,weights=[embedding_matrix_2])(inputB)

y = LSTM(128,return_sequences=True, dropout=0.7)(y)

y = Dense(4, activation="relu")(y)

y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches

combined = concatenate([x.output, y.output])

# apply a FC layer and then a regression prediction on the

# combined outputs

z = Dense(4, activation="relu")(combined)

z = Flatten()(z)

z = Dense(4, activation="linear")(z)

# our model will accept the inputs of the two branches and

# then output a single value

model = Model(inputs=[x.input, y.input], outputs=z)



print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer='adam')

# train the model

print("[INFO] training model...")

model.fit(

    [X_train_s1, X_train_s2], y_train,

    validation_data=([X_validate_s1, X_validate_s2], y_validate),

    epochs=10, batch_size=150)

# make predictions on the testing data

print("[INFO] predicting house prices...")

preds = model.predict([X_test_s1, X_test_s2])
preds

for p in preds:

    print(p)

    print(np.argmax(p))

    break



class_labels = np.argmax(preds, axis=1)

class_labels



proverka = np.argmax(y_test,axis = 1)

proverka
print('accuracy_score = ', end="")

print(accuracy_score(proverka, class_labels))

print('Precision_score = ', end="")

print(precision_score(proverka, class_labels, average='macro'))

print('Recall_score = ', end="")

print(recall_score(proverka, class_labels, average='macro'))

print('F1_score = ', end="")

print(f1_score(proverka, class_labels, average='macro'))
!pip install bert-serving-server
