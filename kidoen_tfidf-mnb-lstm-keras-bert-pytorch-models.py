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
df = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv")
df.head()
df.isnull().sum()
df['text'] = df['Title'] + " " + df['Body']
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['Y'])
df = df.rename(columns={"Y":"label"})
df = df[['text','label','label_enc']]
df.head()
df[['label','label_enc']].drop_duplicates(keep="first")

df.rename(columns={'label':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)
df.rename(columns={"text":"sentence"},inplace=True)
df.head()
import nltk
from nltk.corpus import stopwords
import string
import re
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    return text
df['sentence'] = df['sentence'].apply(clean_text)
df.head()
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def multiclass_logloss(actual, predicted, eps=1e-15):
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df.sentence.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(list(X_train)+list(X_test))
X_train_tfv =  tfv.transform(X_train) 
X_test_tfv = tfv.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_tfv, y_train)
predictions = clf.predict_proba(X_test_tfv)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
# clf = LogisticRegression(C=1.0)
# clf.fit(X_train_tfv, y_train)
# predictions = clf.predict_proba(X_test_tfv)

# print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(X_train) + list(X_test))
X_train_ctv =  ctv.transform(X_train) 
X_test_ctv = ctv.transform(X_test)
# Fitting a simple Logistic Regression on Counts
# clf = LogisticRegression(C=1.0)
# clf.fit(X_train_ctv, y_train)
# predictions = clf.predict_proba(X_test_ctv)

# print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
clf = MultinomialNB()
clf.fit(X_train_ctv, y_train)
predictions = clf.predict_proba(X_test_ctv)
print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))

tokenizer = text.Tokenizer(num_words=None)
max_len = 70
tokenizer.fit_on_texts(list(X_train)+list(X_test))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)
word_index = tokenizer.word_index
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
y_train_enc = np_utils.to_categorical(y_train)
y_test_enc = np_utils.to_categorical(y_test)

import tensorflow
es = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min',
    baseline=None, restore_best_weights=True
)

callbacks = [es]

history = model.fit(X_train_pad,y_train_enc, batch_size=512, epochs=20, verbose=1, validation_data=(X_test_pad, y_test_enc),callbacks=callbacks)
import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
from sklearn.metrics import classification_report,accuracy_score

prediction = model.predict(X_test_pad)

prediction= prediction.argmax(axis=1)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


print("Classification Report : \n",classification_report(y_test, prediction, target_names = ['HQ', 'LQ(Close)', 'LQ(Open)']))
print("\n")
print("Confusion Matrix : \n",confusion_matrix(y_test, prediction))
print("\n")
print("Accuracy Score :",accuracy_score(y_test, prediction))

model = Sequential()
model.add(Embedding(len(word_index)+1,300,input_length=max_len,trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(512, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',restore_best_weights=True)
history = model.fit(X_train_pad, y=y_train_enc, batch_size=512, epochs=20, 
          verbose=1, validation_data=(X_test_pad, y_test_enc), callbacks=[earlystop])

import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
from sklearn.metrics import classification_report,accuracy_score

prediction = model.predict(X_test_pad)

prediction= prediction.argmax(axis=1)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


print("Classification Report : \n",classification_report(y_test, prediction, target_names = ['HQ', 'LQ(Close)', 'LQ(Open)']))
print("\n")
print("Confusion Matrix : \n",confusion_matrix(y_test, prediction))
print("\n")
print("Accuracy Score : ",accuracy_score(y_test, prediction))

