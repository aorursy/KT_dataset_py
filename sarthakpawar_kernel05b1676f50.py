# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def getNonEmpty(t):



    for e in t:

        if e != '':

            return e

    return ''



def NoiseCleansing(DataSet,column):

    Corpus = []

    N = DataSet.shape[0]

    k = 0

    for s in DataSet[column]:

        k += 1

        # 1. Remove ' and " form the description

        s = re.sub('[\'\"]', ' ', s)

        # 2. Remove text inside brackets to get rid of unwanted stuff which can not be labeled



        pat = r'[0-9]+\.[0-9]+'

        qt = re.findall(pat, s)

        if len(qt) > 0:

            for w in qt:

                s = s.replace(w, str(round(float(w))))





        # this regex removes alpha numeric codes

        s = re.sub(r'[a-zA-Z]*[0-9]{4}[a-zA-Z0-9]*', '', s)

        # Remove bad URLS

        s = re.sub(r'http?s:\/\/[a-zA-Z0-9]+[^a-zA-Z0-9]', ' ', s)

        # Remove EMails

        s = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', s)

        # Replace special characters by the space for better tokenization

        s = re.sub(r'[^a-zA-Z0-9\*]+', ' ', s)

        # Remove single char representation of the words

        Removed_Sgl_Char = []

        for w in s.split(' '):

            if re.search(r'[0-9]', w):

                Removed_Sgl_Char.append(w)

            elif (len(w) > 1):

                Removed_Sgl_Char.append(w)



        s = ' '.join(Removed_Sgl_Char)

        Corpus.append(s)

    DataSet[column] = Corpus

    return DataSet
import json

input_data=None

with open('../input/train.json', 'r') as f:

    input_data=json.load(f)
input_data[0]
y=[el['cuisine'] for el in input_data]
input_features=[' '.join([el1 for el1 in el['ingredients']]) for el in input_data]
import pandas as pd 

dataset=pd.DataFrame()
import json

output_data=None

with open('../input/test.json', 'r') as f:

    output_data=json.load(f)
output_data[0]
dataset['text']=input_features

dataset['class']=y
dataset['class'].value_counts()
dataset=dataset.apply(lambda x: x.astype(str).str.lower())
dataset=NoiseCleansing(dataset,'text')
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier
x_train,x_test,y_train,y_test=train_test_split(dataset['text'],dataset['class'],test_size=0.2)
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),

                      ('tfidf', TfidfTransformer()),

                     ('clf', SGDClassifier(loss='modified_huber', penalty='l2',

                                           alpha=1e-4, random_state=42, tol=None)),

 ])

    

ClassiPipelined=OneVsRestClassifier(text_clf)

ClassiPipelined.fit(x_train,y_train)                 

predictedClass=ClassiPipelined.predict(x_test)
import numpy as np
np.mean(predictedClass==y_test)
input_features_test=[' '.join([el1 for el1 in el['ingredients']]) for el in output_data]
ids=[el['id'] for el in output_data]
testset=pd.DataFrame()
testset['id']=ids

testset['text']=input_features_test
testset=testset.apply(lambda x: x.astype(str).str.lower())
testset=NoiseCleansing(testset,'text')
import os

import sys

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Dense, Input, GlobalMaxPooling1D

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

from keras.initializers import Constant
class LanguageIndex():

  def __init__(self):

    self.word2idx = dict()

    self.idx2word = dict()

    self.vocab = set()

  def prepare_mapping(self,text):

    cnt=1

    for t in text:

        for w in t:

            if w not in self.vocab:

                self.word2idx[w]=cnt

                self.idx2word[cnt]=w 

                self.vocab.add(w)

                cnt+=1

  def getIds(self,li):

    return [self.word2idx[el] for el in li]

  def getList(self,li):

    return [self.idx2word[el] for el in li]
LI=LanguageIndex()

LI.prepare_mapping([el['ingredients'] for el in input_data])
input_list=[el['ingredients'] for el in input_data]
# finally, vectorize the text samples into a 2D integer tensor

MAX_NUM_WORDS=len(LI.vocab)+1

sequences=[LI.getIds(el) for el in input_list]

MAX_SEQUENCE_LENGTH=max([len(s) for s in sequences])

# test_sequence=tokenizer.texts_to_sequences(testset['text'])



# word_index = tokenizer.word_index

# print('Found %s unique tokens.' % len(word_index))



data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# data_test=pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

labels = to_categorical(np.asarray(le.fit_transform(dataset['class'])))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
num_words =MAX_NUM_WORDS

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.1



indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-num_validation_samples]

y_train = labels[:-num_validation_samples]

x_val = data[-num_validation_samples:]

y_val = labels[-num_validation_samples:]
embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer='uniform',

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)
len(dataset['class'].unique())
from keras import regularizers
print('Training model.')



# train a 1D convnet with global maxpooling

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(5)(x)

# # x = Conv1D(128, 5, activation='relu')(x)

# # x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

x = Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

preds = Dense(20, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(x)



model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(x_train, y_train,

          batch_size=128,

          epochs=25,validation_data=(x_val,y_val))
from keras import backend as K

from keras.layers import LSTM,Dropout



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
print('Training model.')



# train a 1D convnet with global maxpooling

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x=LSTM(128,activation='tanh',return_sequences=True,kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)

x=Dropout(0.2)(x)

x=LSTM(128,activation='tanh',return_state=True,kernel_regularizer=regularizers.l2(0.01))(x)

x=Dropout(0.2)(x[0])

preds = Dense(20, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=[f1])
model.fit(x_train, y_train,

          batch_size=128,

          epochs=25,validation_data=(x_val,y_val))
y_pred=ClassiPipelined.predict(testset['text'])
del testset['text']

testset['cuisine']=y_pred
testset.to_csv('sample_submission.csv',index=False)