import os
import sys
import re
import pandas as pd
import time
import numpy as np
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
import operator
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
%matplotlib inline


names=['category','title','desc']
df = pd.read_csv('../input/dmoz.csv',delimiter=',',encoding='latin-1',names=names, na_filter=False)
#df.fillna(0)
#df = df[100000:2196860]
#df.drop(df.index[200000:315000], inplace=True)
#df.drop(df.index[810000:925000], inplace=True)
df=df.iloc[1:]
df.head()
df.category.value_counts().plot(figsize=(12,5),kind='bar',color='green')
plt.xlabel('Category')
plt.ylabel('Total Number Of Individual Category')
df.desc = df.title + ' ' + df.desc
df = df.drop(['title'], axis=1)
df.desc = df.desc.str.lower()
lens = [len(x) for x in df.desc]
plt.figure(figsize=(12, 5));
print (max(lens), min(lens), np.mean(lens))
sns.distplot(lens);
plt.title('Description length distribution')
import re
vocab_size = 5100
seq_len = 250
words = [re.findall('[\w\d]+', x) for x in df.desc]
all_words = []
for x in words:
    all_words += x
word_to_id = Counter(all_words).most_common(vocab_size)
word_to_id[:10]
word_to_id[-10:]
word_to_id = {x[0]:i for i, x in enumerate(word_to_id)}
train = [np.array([word_to_id[y] if y in word_to_id else vocab_size-1 for y in x]) for x in words]
train = sequence.pad_sequences(train, maxlen=seq_len, value=0)
train = train.astype('float32')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.category)
y_data = le.transform(df.category)
X_data = df.desc.values
y_data = to_categorical(y_data)
X_train, X_test, y_train, y_test = train_test_split(train, y_data, test_size=0.3, random_state=1000)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_len))
model.add(Dropout(0.1))
model.add(Conv1D(64,5,padding='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128,5,padding='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(64, activation='relu' ))
model.add(Dropout(0.2))
model.add(Dense(13, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.optimizer.lr = 1e-3
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=15,batch_size=64)
from sklearn import metrics
preds = [np.argmax(x) for x in model.predict(X_test)]
y_test_argmax = [np.argmax(x) for x in y_test]
print(metrics.classification_report(preds, y_test_argmax, target_names=le.classes_))
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
array = confusion_matrix(y_test_argmax, preds)
cm=np.array(array)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789ABC"],
                  columns = [i for i in "0123456789ABC"])
plt.figure(figsize = (20,15))
sn.heatmap(df_cm, annot=True)
#model = Sequential()
#model.add(Embedding(max_words,50,input_length=max_len))
#model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(4))
#model.add(Activation('softmax'))
#model.summary()
