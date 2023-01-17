# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir('../input')

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import nltk
import os
import gc
import time
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import re
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer

start_sec = time.time()

train = pd.read_csv('train.tsv', sep='\t')
train['Sentiment'] += 2
test = pd.read_csv('test.tsv', sep='\t')
test['Sentiment'] = -999
df = pd.concat([train, test], ignore_index = True)

lemma = WordNetLemmatizer()
clean_phrase = []
for i in range(len(df['Phrase'])):
  t = re.sub('[^a-z]', ' ', df['Phrase'][i].lower())
  temp = [lemma.lemmatize(word) for word in word_tokenize(t)]
  seq = ' '.join(temp)
  clean_phrase.append(seq)


#print(train.head())
df['clean'] = clean_phrase

train_data = df[df['Sentiment']!=-999]
test_data = df[df['Sentiment']==-999]
train_phrase = df[df['Sentiment']!=-999]['clean']
test_phrase = df[df['Sentiment']==-999]['clean']
#del df

train_label = to_categorical(train_data['Sentiment'])

train_x, valid_data, label_x, valid_label = train_test_split(train_data['Phrase'], train_label, test_size = 0.2)


all_words = word_tokenize(' '.join(train_phrase))
dist = FreqDist(all_words)
num_words = len(dist)
l = []
for i in train_data['clean']:
  l.append(len(word_tokenize(i)))

max_seq_length = max(l)

max_features = num_words
max_words = max_seq_length
batch_size = 128
epochs = 10
num_classes=train_label.shape[1]

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x))
X_train = tokenizer.texts_to_sequences(list(train_x))
X_val = tokenizer.texts_to_sequences(list(valid_data))
X_test = tokenizer.texts_to_sequences(list(test_phrase))

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


model = Sequential()
model.add(Embedding(max_features,250,mask_zero=True))

model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(32))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()

history=model.fit(X_train, label_x, validation_data=(X_val, valid_label),epochs=5, batch_size=batch_size, verbose=1)
print('running time is ', time.time()-start_sec)
y_pred = model.predict_classes(X_test)
sub=pd.read_csv('sampleSubmission.csv')
sub.Sentiment=y_pred
#f = open('sub.csv', 'w')
sub.to_csv('submission.csv',index=False)