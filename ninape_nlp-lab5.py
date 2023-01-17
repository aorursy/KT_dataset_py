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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

from six.moves import urllib

import zipfile

import lxml

from lxml import etree

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.text import one_hot

from keras.layers import Embedding

from keras.layers import Dense, Activation, Flatten, Masking

from keras.layers.recurrent import LSTM

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
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
def load_data():

    if not os.path.isfile('data/ted_en-20160408.zip'):

        urllib.request.urlretrieve('https://wit3.fbk.eu/get.php?path='

                                   'XML_releases/xml/'

                                   'ted_en-20160408.zip&filename=' 

                                   'data/ted_en-20160408.zip', filename='ted_en-20160408.zip')

    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z: 

        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

    text = [str(t) for t in doc.xpath('//content/text()')] 

    keywords = [str(t) for t in doc.xpath('//keywords/text()')] 

    return text, keywords
glove = loadGloveModel('/kaggle/input/lab4-nlp/glove.6B.50d.txt')
text , keywords = load_data()
print(text[0])

print(keywords[0])
df = pd.DataFrame(columns=['text', 'keywords', 'class'])

print(df)
c = []

for t, k in zip(text, keywords):

    if 'technology' in k and 'design' in k and 'entertainment' in k :

        m ={'text':t,'keywords':k,'class':'TED'}

        df = df.append(m, ignore_index=True)

        c.append('TED')

    elif 'entertainment' in k and 'design' in k:

        m = {'text':t,'keywords':k,'class':'oED'}

        df = df.append(m, ignore_index=True)

        c.append('oED')

    elif 'entertainment' in k and 'technology' in k:

        m = {'text':t,'keywords':k,'class':'TEo'}

        df = df.append(m, ignore_index=True)

        c.append('TEo')

    elif 'technology' in k and 'design' in k:

        m = {'text':t,'keywords':k,'class':'ToD'}

        df = df.append(m, ignore_index=True)

        c.append('ToD')

    elif 'technology' in k:

        m = {'text':t,'keywords':k,'class':'Too'}

        df = df.append(m, ignore_index=True)

        c.append('Too')

    elif 'entertainment' in k:

        m = {'text':t,'keywords':k,'class':'oEo'}

        df = df.append(m, ignore_index=True)

        c.append('oEo')

    elif 'design' in k:

        m = {'text':t,'keywords':k,'class':'ooD'}

        df = df.append(m, ignore_index=True)

        c.append('ooD')    

    else:

        m = {'text':t,'keywords':k,'class':'ooo'}

        df = df.append(m, ignore_index=True)

        c.append('ooo')
print(len(c))

print(len(text))
print('ooo',c.count('ooo'))

print('Too',c.count('Too'))

print('oEo',c.count('oEo'))

print('ooD',c.count('ooD'))

print('TEo',c.count('TEo'))

print('ToD',c.count('ToD'))

print('oED',c.count('oED'))

print('TED',c.count('TED'))

print(df)
label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(c)

print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
w = []

for k in keywords:

    k = k.split(',')

    #print(k)

    for word in k:

        word = word.replace(' ','')

        #print(word)

        #if word is not ',':

        w.append(word)

    #break

unique_w = set(w)

vocab_length = len(unique_w)

print(vocab_length)

print(len(w))

print(unique_w)
seq = []

for k in keywords:

    k = k.replace(',', '')

    #print(k)

    v= vocab_length+10

    oh = one_hot(k, vocab_length)

    #print(oh)

    seq.append(oh)

print(seq[0:2])

print(keywords[0:2])

    
m = 0

for s in seq:

    #print(s)

    #print(len(s))

    if len(s) > m:

        m = len(s)

print(m)
padded_seq = pad_sequences(seq, m, padding='post')

print(padded_seq)
X_train, X_test, y_train, y_test = train_test_split(padded_seq, onehot_encoded, test_size=0.2, random_state=1)



X_train = np.array(X_train)

y_train = np.array(y_train)



X_test = np.array(X_test)

y_test = np.array(y_test)



print(X_train.shape)

print(y_train.shape)
from keras.regularizers import l2

from keras.layers.recurrent import GRU

from keras.layers import Bidirectional, TimeDistributed

from keras.layers import Dropout

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D



#bez glove

model = Sequential()

model.add(Embedding(vocab_length+1, 20, input_length = 32))

#model.add(Masking(mask_value=0.0))

#model.add(GRU(64)) #dropout=0.1, recurrent_dropout=0.1))

#model.add(Bidirectional(LSTM(64))) 

#model.add(LSTM(64))

model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(1668,32)))

model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))

model.add(LSTM(64))

model.add(Dense(256, activation='relu'))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dense(512, activation='relu'))

model.add(Dense(8, activation='softmax'))





model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
m1 = model.fit(X_train, y_train, epochs=15, batch_size=30)
predictions = model.predict(X_test)

print(predictions[0:5])
y_new = list()

for i in predictions:

    max = 0

    index = 0

    for j in range(0, 8):

        if i[j] > max:

            max = i[j]

            index = j

    y_new.append(index)



y_test_new = list()

for i in y_test:

    max = 0

    index = 0

    for j in range(0, 8):

        if i[j] > max:

            max = i[j]

            index = j

    y_test_new.append(index)

    

#print(y_new)

#print(y_test_new)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_new, y_new)

print(cm)

print('Accuracy = ', end="")

print((cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7])/len(y_new))



print('Precision_score = ', end="")

print(precision_score(y_test_new, y_new, average='macro'))

print('Recall_score = ', end="")

print(recall_score(y_test_new, y_new, average='macro'))

print('F1_score = ', end="")

print(f1_score(y_test_new, y_new, average='macro'))
ks = []

for k in keywords:

    k = k.replace(',','')

    #print(k)

    #break

    ks.append(k)

print (ks)
from keras.preprocessing.text import Tokenizer

word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(ks)

vocab_length = len(word_tokenizer.word_index) + 1

embedded_sentences = word_tokenizer.texts_to_sequences(ks) # gi preveduvam vo numericki vrednosti

#print(embedded_sentences)

m = 0

for s in embedded_sentences:

    #print(s)

    #print(len(s))

    if len(s) > m:  #go baram najdolgiot za da gi padnuvam site do tamu

        m = len(s)

length_long_sentence = m

padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')

print(padded_sentences)
#sega model so glove

embedding_matrix = np.zeros((vocab_length, 50))

for word, index in word_tokenizer.word_index.items():

    if word in glove:

        embedding_vector = glove[word]

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
#sega model so glove

from keras.regularizers import l2

from keras.layers.recurrent import GRU

from keras.layers import Bidirectional

from keras.layers import Dropout

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D





modelGlove = Sequential()

embedding_layer = Embedding(vocab_length, 50, weights=[embedding_matrix], input_length= 32, trainable=False)

modelGlove.add(embedding_layer)

#modelGlove.add(GRU(64)) #dropout=0.1, recurrent_dropout=0.1))

#modelGlove.add(Bidirectional(LSTM(64))) 

modelGlove.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(1668,32)))

modelGlove.add(Conv1D(filters=64, kernel_size=5, activation='relu'))

modelGlove.add(LSTM(64))

modelGlove.add(Dense(512, activation='relu'))

modelGlove.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

modelGlove.add(Dense(256, activation='relu'))

modelGlove.add(Dense(8, activation='softmax'))





modelGlove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
m2 = modelGlove.fit(X_train, y_train, epochs=15, batch_size=30)
predictionsGlove = modelGlove.predict(X_test)

print(predictionsGlove[0:5])
y_new_g = list()

for i in predictionsGlove:

    max = 0

    index = 0

    for j in range(0, 8):

        if i[j] > max:

            max = i[j]

            index = j

    y_new_g.append(index)



y_test_new_g = list()

for i in y_test:

    max = 0

    index = 0

    for j in range(0, 8):

        if i[j] > max:

            max = i[j]

            index = j

    y_test_new_g.append(index)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_new_g, y_new_g)

print(cm)

print('Accuracy = ', end="")

print((cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7])/len(y_new_g))



print('Precision_score = ', end="")

print(precision_score(y_test_new_g, y_new_g, average='macro'))

print('Recall_score = ', end="")

print(recall_score(y_test_new_g, y_new_g, average='macro'))

print('F1_score = ', end="")

print(f1_score(y_test_new_g, y_new_g, average='macro'))
# Get loss histories

m1_loss = m1.history['loss']

m2_loss = m2.history['loss']

epoch_count = range(1, len(m1_loss) + 1)

plt.plot(epoch_count, m1_loss, 'r--')

plt.plot(epoch_count, m2_loss, 'b-')

plt.legend(['M1 Loss', 'M2 Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()