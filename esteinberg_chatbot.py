# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras as keras
from keras import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from keras.layers import TimeDistributed 
from keras.layers import Conv1D 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!python -m spacy download fr_core_news_lg
"""
!python -m spacy download fr_core_news_sm
import fr_core_news_lg
nlp = fr_core_news_lg.load()
nlp.to_disk('/kaggle/working/fr_lg')
"""
import fr_core_news_lg
nlp = fr_core_news_lg.load()
#nlp = spacy.load('/kaggle/working/fr_lg')
df = pd.read_csv('https://raw.githubusercontent.com/EzrielS/datasets/master/chatbot_imprimante.csv', )
df=df.drop(columns='Unnamed: 0')
df.text = df.text.map(str.lower)
df.doc = df.doc.map(lambda x: x.lower() if not pd.isna(x) else np.nan)
df
# define example
values = [' '.join(j.lemma_ for j in nlp(i)) for i in df.text]
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("il y a ", len(label_encoder.classes_)," mots différents")
print(integer_encoded)
# binary encode
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
class LabelEncoderExt(object):
    def __init__(self, unknown_val='Unknown'):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        self.unknown_val = unknown_val
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + [self.unknown_val])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = [self.unknown_val if x==unique_item else x for x in new_data_list]
        print (new_data_list)
        return self.label_encoder.transform(new_data_list)
def getLemmasEncoder(txts):
    flat_list = [item.lemma_ for sublist in txts for item in nlp(sublist)]
    le = LabelEncoderExt()
    le.fit(flat_list)
    return le

def getTextsAsLemmas(txts):
    return [[i.lemma_ for i in nlp(L)] for L in txts]
le= getLemmasEncoder(df.text[0:10])
# define example
values = [' '.join(j.lemma_ for j in nlp(i)) for i in df.text]
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("il y a ", len(label_encoder.classes_)," mots différents")
print(integer_encoded)
# binary encode
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
TAILLE_PHRASE_MAX = 50
LEN_VECTOR = len(nlp("bonjour")[0].vector)
TYPES = ['ADJ','ADP','ADV','AUX','CONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X','SPACE','CCONJ']
LEN_TYPES = len(TYPES)
def typeAsList(pos):
    res = [0.] * LEN_TYPES
    if pos in TYPES:
        res[TYPES.index(pos)]=1.
    else:
        print (pos)
    return res

def textsToArray(txts):
    return  np.array([
        np.array(
            [np.array(typeAsList(i.pos_) + list(i.vector))
                 for i in nlp(phrase)] + 
            [np.zeros((LEN_VECTOR+LEN_TYPES)) for i in range(TAILLE_PHRASE_MAX-len(nlp(phrase)))])
        for phrase in txts
    ])

res = textsToArray(df.text)
!pip install livelossplot
from livelossplot import PlotLossesKeras

res.shape

xtrain, xtest, ytrain, ytest = train_test_split(res, df.correct,test_size=.2)

nn = Sequential()
nn.add(Masking(0, input_shape=(None, LEN_VECTOR+LEN_TYPES)))
nn.add(Bidirectional(LSTM(4,
                         dropout=.4,
                       recurrent_dropout=.1,
                         )))
nn.add(Dense(1, activation='sigmoid'))
nn.compile(optimizer="adam", 
           loss='binary_crossentropy',  
           metrics=['acc'])
nn.build()
nn.summary()

nn.fit(xtrain, ytrain, 
       epochs=1000, 
       callbacks=[
            PlotLossesKeras(), 
            EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
       ],
       validation_data=(xtest, ytest), 
        shuffle=True,
#        batch_size=16
      )
nn.save("ModelPhraseCorrecte")
res[0][0].shape
pd.DataFrame(ytest)[(~ (nn.predict_classes(xtest)[:,0] == ytest))]
df.loc[52]

ytest.mean()

confusion_matrix(ytest, nn.predict_classes(xtest),  )
nn.predict_classes(xtest)
getTextsAsLemmas(df.text)

def phraseToOneHotDocument(phr, doc_name):
    nlped = nlp(phr)
    res = np.zeros(len(nlped))
    for nb,i in enumerate(nlped):
        if i.text == doc_name:
            res[nb]=1.
            break
    return res
    
phraseToOneHotDocument(df.text[2],df.doc[2] )
df.text[2]
data_doc = df[df.correct==1]
data_doc
data_doc['doc_vector'] = [phraseToOneHotDocument(i.text,i.doc) for _,i in data_doc.iterrows()]
data_doc
xtrain, xtest, ytrain, ytest = train_test_split(res, df.correct,test_size=.2)


nn = Sequential()
nn.add(Masking(0, input_shape=(None, LEN_VECTOR)))
nn.add(Bidirectional(LSTM(3, return_sequences=True)))
nn.add(TimeDistributed(Dense(1, activation='sigmoid')))
nn.compile(optimizer="adam", 
           loss='mse',  
#           metrics=['acc']
          )
nn.build()
nn.summary()
xtrain, xtest, ytrain, ytest = train_test_split(res, df.correct,test_size=.2)
nn = Sequential()
nn.add(Conv1D(1, 
              kernel_size=5, 
              activation='sigmoid', 
              input_shape=(None, TAILLE_PHRASE_MAX, LEN_VECTOR),
              padding='same',
             ))
nn.compile(optimizer="adam", 
           loss='mse',
           
#           metrics=['acc']
          )
nn.build()
nn.summary()
correctsArr = textsToArray(data_doc.text)
tr = data_doc.doc_vector.map(lambda x: x+.0001)
trr = np.array([list(i) + [0]*(TAILLE_PHRASE_MAX-len(i)) for i in tr])
trr=trr.reshape((len(trr),-1,1))
xtr, xte, ytr, yte = train_test_split(correctsArr, trr)
correctsArr.shape
nn.fit(xtr,
       ytr,
       epochs=5000,
#       validation_split=0.3,
       validation_data=(xte, yte),
       callbacks=[PlotLossesKeras(),
                  EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)],
      )
nn.predict(xtr[0:5])-ytr[0:5]
nn.fit(correctsArr,
       trr,
       epochs=1000,
       callbacks=[PlotLossesKeras()],
       validation_split=.3
      )
xtr.shape
yte[0]
data_doc.loc[15]
data_doc.loc[20].text
trr[0]
correctsArr.shape
X.shape

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)
y.shape
X.shape
X

y
