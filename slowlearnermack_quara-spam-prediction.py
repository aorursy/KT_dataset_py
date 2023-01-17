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

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers.convolutional import Conv1D

from keras.layers import MaxPooling1D , GlobalMaxPooling1D
!wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip the file 

!unzip glove*.zip
#store 300 size vector representation different words from the file to a disctionary

embedding_index = {}

f = open('glove.6B.300d.txt',encoding='utf-8')

for line in f:

  value = line.split()

  word = value[0]

  coeffs = np.asarray(value[1:],dtype = 'float32')

  embedding_index[word] = coeffs

f.close()
len(embedding_index)
df = pd.read_csv('../input/train.csv')
df.head()
df.tail()
print(df['target'].value_counts())

sns.countplot(df['target'])
#to check the proportion of the ones to zeros

target_count = df['target'].value_counts()



print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');
x = df['question_text']

y = df['target']
token = Tokenizer()
from sklearn.model_selection import train_test_split

q_train , q_test = train_test_split(df, test_size=0.2)
q_train.shape , q_test.shape
x_train = q_train['question_text']

y_train = q_train['target']

x_test  = q_test['question_text']

y_test  = q_test['target']
x_train.shape , y_train.shape , x_test.shape , y_test.shape
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)

class_weights
token.fit_on_texts(x)

seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq,maxlen=300)
vocab_size = len(token.word_index)+1
x = df['question_text']

y = df['target']
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.layers import LSTM,Activation,Dense,Input,Embedding,Dropout

from keras.models import Model

from nltk import word_tokenize

import nltk

nltk.download('punkt')
sent_lens=[len(word_tokenize(x)) for x in x_train]
max(sent_lens)
np.percentile(sent_lens,95)
max_len=31 #taking the 95% quantile value of the sentence length



tk=Tokenizer(char_level=False,split=' ') # tokenizing the sentence 



tk.fit_on_texts(x_train)



seq_train=tk.texts_to_sequences(x_train) # create tokens on train

seq_test=tk.texts_to_sequences(x_test) # create tokens on test



vocab_size=len(tk.word_index)



seq_train_matrix=sequence.pad_sequences(seq_train,maxlen=max_len) #padding the sentence with 0 for matching length

seq_test_matrix=sequence.pad_sequences(seq_test,maxlen=max_len)
seq_train_matrix.shape , seq_test_matrix.shape , vocab_size
seq_train_matrix[1]
# creating our own embedding matrix to bring down the size to 300

# we'll use 300 D vector representation of the words from pretrained embedding index 

# that we downloaded 



embedding_matrix=np.zeros((vocab_size+1,300))



for word,i in tk.word_index.items():

    embed_vector=embedding_index.get(word)

    if embed_vector is not None:

        embedding_matrix[i]=embed_vector

# if there are specific words which are not present in pretrained embedding 

# their weights will remain 0. if there are too many such words 

# then you should probably not use pretrained embeddings
inputs=Input(name='text_input',shape=[max_len])

embed=Embedding(vocab_size+1,300,input_length=max_len,mask_zero=True,

                weights=[embedding_matrix],trainable=False)(inputs)



GRU_layer=GRU(50)(embed)



dense1=Dense(10,activation='relu')(GRU_layer)

drop=Dropout(0.2)(dense1)



final_layer=Dense(1,activation='sigmoid')(drop)



model_GRU=Model(inputs=inputs,outputs=final_layer)

model_GRU.summary()
model_GRU.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

import os

outputFolder = './content/Model_output/'

if not os.path.exists(outputFolder):

    os.makedirs(outputFolder)

filepath = outputFolder+"/weights-{epoch:02d}-{val_acc:.4f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 

                             save_best_only=False, save_weights_only=True, 

                             mode='auto', period=1)

# this will save the weights every 10 epoch



from keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3,

                          verbose=1, mode='auto')
model_GRU.fit(seq_train_matrix,y_train,validation_data=[seq_test_matrix,y_test],epochs=10,class_weight={0:0.53,1:8},

          batch_size=10000,callbacks=[earlystop,checkpoint])
p=model_GRU.predict(seq_test_matrix)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,p)
from sklearn.metrics import classification_report,f1_score

print(f1_score(y_test,p >.50))
print(classification_report(y_test,p>.50))