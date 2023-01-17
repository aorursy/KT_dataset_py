# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/quora-question-pairs/train.csv.zip')
df.head()
test_data = pd.read_csv('../input/quora-question-pairs/test.csv.zip')
test_data.head()
X_train = df.iloc[:,:5].values

Y_train = df.iloc[:,5:].values
X_testq1 = test_data.iloc[:400001,1:2].values

X_testq2 = test_data.iloc[:400001, 2:].values
s1 = X_train[:,3]

s2 = X_train[:,4]
def tokenize(s):

    tokens = []

    tokens = [word_tokenize(str(sentence)) for sentence in s]



    rm1 = []

    for w in tokens:

        sm = re.sub('[^A-Za-z]',' ', str(w))

        x = re.split("\s", sm)

        rm1.append(x)

        

    return rm1
def lower_case(s):

    #Removing whitespaces    

    for sent in s:

        while '' in sent:

            sent.remove('')



    # Lowercasing

    low = []

    for i in s:

        i = [x.lower() for x in i]

        low.append(i)

        

    return low

    
def lemmatize(s):

    lemma = []

    wnl = WordNetLemmatizer()

    for doc in s:

        tokens = [wnl.lemmatize(w) for w in doc]

        lemma.append(tokens)



    # Removing Stopwords

    filter_words = []

    Stopwords = set(stopwords.words('english'))



    #ab = spell('nd')

    for sent in lemma:

        tokens = [w for w in sent if w not in Stopwords]

        filter_words.append(tokens)



    space = ' ' 

    sentences = []

    for sentence in filter_words:

        sentences.append(space.join(sentence))

        

    return sentences
# sent1 = tokenize(s1)

# sent2 = tokenize(s2)

# q1 = lower_case(sent1)

# q2 = lower_case(sent2)

# listq1 = lemmatize(q1)

# listq2 = lemmatize(q2)

# sent1_t = tokenize(X_test_q1)

# sent2_t = tokenize(X_test_q2)

# q1_t = lower_case(sent1_t)

# q2_t = lower_case(sent2_t)

# listq1 = lemmatize(q1_t)

# listq2 = lemmatize(q2_t)
MAX_NB_WORDS = 200000

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)

tokenizer.fit_on_texts(list(df['question1'].values.astype(str))+list(df['question2'].values.astype(str)))

# X_train_q1 = tokenizer.texts_to_sequences(np.array(listq1))

X_train_q1 = tokenizer.texts_to_sequences(df['question1'].values.astype(str))

X_train_q1 = pad_sequences(X_train_q1, maxlen = 30, padding='post')



# X_train_q2 = tokenizer.texts_to_sequences(np.array(listq2))

X_train_q2 = tokenizer.texts_to_sequences(df['question2'].values.astype(str))

X_train_q2 = pad_sequences(X_train_q2, maxlen = 30, padding='post')

X_test_q1 = tokenizer.texts_to_sequences(X_testq1.ravel())

X_test_q1 = pad_sequences(X_test_q1,maxlen = 30, padding='post')



X_test_q2 = tokenizer.texts_to_sequences(X_testq2.astype(str).ravel())

X_test_q2 = pad_sequences(X_test_q2, maxlen = 30, padding='post')
word_index = tokenizer.word_index
embedding_index = {}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:], 'float32')

        embedding_index[word] = vectors

    f.close()
embedding_matrix = np.random.random((len(word_index)+1, 200))

for word, i in word_index.items():

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# Model for Q1

import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization

model_q1 = tf.keras.Sequential()

model_q1.add(Embedding(input_dim = len(word_index)+1,

                       output_dim = 200,

                      weights = [embedding_matrix],

                      input_length = 30))

model_q1.add(LSTM(128, activation = 'tanh', return_sequences = True))

model_q1.add(Dropout(0.2))

model_q1.add(LSTM(128, return_sequences = True))

model_q1.add(LSTM(128))

model_q1.add(Dense(60, activation = 'tanh'))

model_q1.add(Dense(2, activation = 'sigmoid'))
# Model for Q2

model_q2 = tf.keras.Sequential()

model_q2.add(Embedding(input_dim = len(word_index)+1,

                       output_dim = 200,

                      weights = [embedding_matrix],

                      input_length = 30))

model_q2.add(LSTM(128, activation = 'tanh', return_sequences = True))

model_q2.add(Dropout(0.2))

model_q2.add(LSTM(128, return_sequences = True))

model_q2.add(LSTM(128))

model_q2.add(Dense(60, activation = 'tanh'))

model_q2.add(Dense(2, activation = 'sigmoid'))
# Merging the output of the two models,i.e, model_q1 and model_q2

mergedOut = Multiply()([model_q1.output, model_q2.output])



mergedOut = Flatten()(mergedOut)

mergedOut = Dense(100, activation = 'relu')(mergedOut)

mergedOut = Dropout(0.2)(mergedOut)

mergedOut = Dense(50, activation = 'relu')(mergedOut)

mergedOut = Dropout(0.2)(mergedOut)

mergedOut = Dense(2, activation = 'sigmoid')(mergedOut)
new_model = Model([model_q1.input, model_q2.input], mergedOut)

new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',

                 metrics = ['accuracy'])

history = new_model.fit([X_train_q1,X_train_q2],Y_train, batch_size = 2000, epochs = 10)
y_pred = new_model.predict([X_test_q1, X_test_q2], batch_size=2000, verbose=1)

y_pred += new_model.predict([X_test_q1, X_test_q2], batch_size=2000, verbose=1)

y_pred /= 2