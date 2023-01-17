import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,BatchNormalization
from keras.utils import to_categorical
dftrain = pd.read_csv('../input/train.tsv',sep='\t')
dftrain.head()
dftrain.describe()
model_vec = CountVectorizer(stop_words='english',min_df=30,ngram_range=(2,4)).fit(dftrain['Phrase'])
print(len(model_vec.get_feature_names()))
df = pd.DataFrame(model_vec.transform(dftrain['Phrase']).toarray())
df.columns = model_vec.get_feature_names()
print(df.shape)
df.head()
if 1==2:
    x_train = np.array(df.iloc[:1000,:].copy()).reshape(1000,1,df.shape[1])
    y_train = np.array(dftrain.loc[:999,'Sentiment'].copy()).reshape(1000,1)

    y_train = to_categorical(y_train)
    print(x_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(1,df.shape[1])))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, nb_epoch=10,validation_split=0.3)
x_train = np.array(df)
y_train = np.array(dftrain['Sentiment'].copy())
y_train = to_categorical(y_train)
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(500, activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dropout(rate=0.5))
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Nadam',metrics=['accuracy']) 
print(model.summary())
model.fit(x_train,y_train, epochs=10, validation_split=0.5)
del [df,x_train,y_train,dftrain]
dftest = pd.read_csv('../input/test.tsv',sep='\t')
df = pd.DataFrame(model_vec.transform(dftest['Phrase']).toarray())
df.columns = model_vec.get_feature_names()
print(df.shape)
x_test = np.array(df)
dfout = model.predict(x_test)
dfout = pd.DataFrame(dfout).round(2)
dfout.head()
dfout = pd.DataFrame({'PhraseId':dftest.PhraseId,'Sentiment':dfout.idxmax(axis=1)})
dfout.describe()
dfout.to_csv('submission.csv',index=False)