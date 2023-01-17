# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/full_dataset.csv',sep=';')

df.head()
df.info()
## importing everything that we're going to need!



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, LSTM, SimpleRNN

from keras.layers.wrappers import Bidirectional

from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical



from gensim import corpora



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



import re
def replace_tags(list_of_tokens: list) -> list:

    

    treated_tokens = []

    for token in list_of_tokens:

        if token == '?':

            treated_tokens.append('<QST_MARK>')

        elif token == '!':

            treated_tokens.append('<EXC_MARK>')

        else:

            treated_tokens.append(token)

            

    return treated_tokens 



def pad_sequence(list_of_treated_tokens: list, max_len: int) -> list:

    

    actual_len = len(list_of_treated_tokens)

    if actual_len < max_len:

        padded_list = ['<PAD>' for _ in range(max_len-actual_len)] + list_of_treated_tokens

    elif actual_len > max_len:

        padded_list = list_of_treated_tokens[:max_len]

    else:

        padded_list = list_of_treated_tokens

        

    return padded_list

    

def pre_processing(text: str) -> str:

    text = re.sub('([!?])',r' \1 ',text)

    text = text.lower().split()

    

    ## adding tokens

    list_with_tokens = replace_tags(text)

    

    ## padding the sentence

    padded_list = pad_sequence(list_with_tokens,max_len=30)

       

    return padded_list
## applying pre_processing



df['text_tratado'] = df.title.apply(pre_processing)
## encoding using a simple bow model



encoder = corpora.Dictionary(df.text_tratado)



x = np.array([encoder.doc2idx(s) for s in df.text_tratado])
## defining y

y = df.label

y_ann = to_categorical(y.map({'python':0,'java':1,'R':2,'javascript':3,'php':4}),5)
## train test



x_train,x_valid,y_train,y_valid = train_test_split(x,y_ann,test_size=0.2)
cross_val_score(SVC(),x,y,cv=10)
cross_val_score(RandomForestClassifier(500),x,y,cv=10)
## defining the model



model = Sequential()

model.add(Embedding(x.shape[1],64))

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(LSTM(256, dropout=0.2)))

model.add(Dense(32,activation= 'relu'))

model.add(Dense(16,activation= 'relu'))

model.add(Dense(8,activation= 'relu'))

model.add(Dense(5, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(x_train,y_train,batch_size = 32, epochs = 10, validation_data = (x_valid,y_valid))
model2 = Sequential()

model2.add(Embedding(x.shape[1],64))

model2.add(SpatialDropout1D(0.2))

model2.add(SimpleRNN(256, dropout=0.2))

model2.add(Dense(32,activation= 'relu'))

model2.add(Dense(16,activation= 'relu'))

model2.add(Dense(8,activation= 'relu'))

model2.add(Dense(5, activation='softmax'))



model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.summary()
model2.fit(x_train,y_train,batch_size = 32, epochs = 10, validation_data = (x_valid,y_valid))