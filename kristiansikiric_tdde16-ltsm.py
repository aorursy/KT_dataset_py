# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv("/kaggle/input/tdde16-preprocessing/processed_lyrics_genres.csv")

df = df.dropna()



print(df.head())

print(df["Genre"].value_counts())

df["Genre"].value_counts().plot(kind='bar')



from imblearn.over_sampling import RandomOverSampler

X = df.drop('Genre',axis=1)

y = df["Genre"]

sm = RandomOverSampler(random_state = 123)

X_sampled,y_sampled = sm.fit_resample(X,y)



df_oversampled = pd.concat([pd.DataFrame(X_sampled,columns = ["Lyrics"]), pd.DataFrame(y_sampled, columns=['Genre'])], axis=1)

print(df_oversampled["Genre"].value_counts())

df_oversampled["Genre"].value_counts().plot(kind='bar')



df_train = df_oversampled.sample(frac = 0.8,random_state = 123)

df_test = df_oversampled.drop(df_train.index)



del df

gc.collect()



genres = df_test.Genre.unique()



# Any results you write to the current directory are saved as output.
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence



max_features = 20000

maxlen = 80

nb_classes = 5

embedding_dims = 50

filters = 250

kernel_size = 3

hidden_dims = 250



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(df_train['Lyrics'])

sequences_train = tokenizer.texts_to_sequences(df_train['Lyrics'])

sequences_test = tokenizer.texts_to_sequences(df_test['Lyrics'])



X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)

X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping



from keras.models import Sequential

from keras.layers import LSTM,Dropout,Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers.embeddings import Embedding



def create_model():

    model = Sequential()

    model.add(Embedding(max_features, 128))

    

    model.add(Dropout(0.2))

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(5, activation='softmax'))



    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])



    return model



model = KerasClassifier(build_fn=create_model,verbose=0, epochs = 300,

                        callbacks = [EarlyStopping(patience=30, monitor = 'val_loss',restore_best_weights=True)],

                        validation_split=0.2,batch_size = 1000)



pipeline = Pipeline(steps=[

    #('vect', CountVectorizer()),

    ('keras', model)

])



pipeline.fit(X_train,df_train["Genre"])

y_pred = pipeline.predict(X_test)

y_true = df_test["Genre"]

print(classification_report(y_true, y_pred))

print(confusion_matrix(y_true,y_pred,labels=genres))

print(genres)



del y_pred

del y_true

gc.collect()