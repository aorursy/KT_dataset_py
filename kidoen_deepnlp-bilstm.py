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
import pandas as pd
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
train = pd.read_csv('/kaggle/input/spooky-author-identification/train.zip')
test = pd.read_csv('/kaggle/input/spooky-author-identification/test.zip')
sample = pd.read_csv('/kaggle/input/spooky-author-identification/sample_submission.zip')
train.head()
train.isnull().sum()
test.head()
sample.head()
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
labelencoder = preprocessing.LabelEncoder()
y = labelencoder.fit_transform(train.author.values)
y
X_train, X_test, y_train, y_test = train_test_split(train.text.values, y, 
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
X_train_tfv
clf = LogisticRegression(C=1.0)
clf.fit(X_train_tfv, y_train)
predictions = clf.predict_proba(X_test_tfv)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(X_train) + list(X_test))
X_train_ctv =  ctv.transform(X_train) 
X_test_ctv = ctv.transform(X_test)
# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(X_train_ctv, y_train)
predictions = clf.predict_proba(X_test_ctv)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
clf = MultinomialNB()
clf.fit(X_train_tfv, y_train)
predictions = clf.predict_proba(X_test_tfv)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
clf = MultinomialNB()
clf.fit(X_train_ctv, y_train)
predictions = clf.predict_proba(X_test_ctv)
print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))

svd = decomposition.TruncatedSVD(n_components=150)
svd.fit(X_train_tfv)
X_train_svd = svd.transform(X_train_tfv)
X_test_svd = svd.transform(X_test_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(X_train_svd)
X_train_svd_scl = scl.transform(X_train_svd)
X_test_svd_scl = scl.transform(X_test_svd)
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(X_train_svd_scl, y_train)
predictions = clf.predict_proba(X_test_svd_scl)
print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))

X_train
tokenizer = text.Tokenizer(num_words=None)
max_len = 70
tokenizer.fit_on_texts(list(X_train)+list(X_test))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)
word_index = tokenizer.word_index
X_train_pad
len(word_index)
word_index

    
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
y_test.shape
y_test_enc.shape

model.fit(X_train_pad,y_train_enc, batch_size=512, epochs=100, verbose=1, validation_data=(X_test_pad, y_test_enc))
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(X_train_pad, y=y_train_enc, batch_size=512, epochs=500, 
          verbose=1, validation_data=(X_test_pad, y_test_enc), callbacks=[earlystop])
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
model.fit(X_train_pad, y=y_train_enc, batch_size=512, epochs=500, 
          verbose=1, validation_data=(X_test_pad, y_test_enc), callbacks=[earlystop])
