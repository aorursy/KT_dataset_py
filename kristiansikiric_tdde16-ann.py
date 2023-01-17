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
from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



def create_model():

    model = Sequential()

    model.add(Dense(10,input_shape=(441070,), activation='relu'))

    model.add(Dense(5, activation='softmax'))



    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])



    return model



model = KerasClassifier(build_fn=create_model,verbose=0, epochs = 150,

                        callbacks = [EarlyStopping(patience=10, monitor = 'val_loss',restore_best_weights=True)],

                        validation_split=0.2, batch_size = 500)



pipeline = Pipeline(steps=[

    ('vect', CountVectorizer()),

    ('keras', model)

])

pipeline.fit(df_train["Lyrics"],df_train["Genre"])

y_pred = pipeline.predict(df_test["Lyrics"])

y_true = df_test["Genre"]

print(classification_report(y_true, y_pred))

print(confusion_matrix(y_true,y_pred,labels=genres))

print(genres)



del y_pred

del y_true

gc.collect()