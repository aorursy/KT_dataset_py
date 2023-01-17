# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, ConvLSTM2D, SpatialDropout1D,Bidirectional, Dropout, Conv1D, MaxPooling1D

import keras

import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print(df_train.head(10))

print(df_test.head(10))
df_train.describe()
df_train.info()
df_train.shape


url = re.compile(r'https?://\S+|www\.\S+')

train = df_train['text'].apply(lambda tweet: url.sub(r'',tweet))

test = df_test['text'].apply(lambda tweet: url.sub(r'',tweet))



print(train)
train=train.str.lower().str.replace("[^a-z]", " ")

test=test.str.lower().str.replace("[^a-z]", " ")




lem = WordNetLemmatizer()



stop_words = stopwords.words('english')



l1 = []

for i in train:

    temp = word_tokenize(i)

    l1.append(" ".join(lem.lemmatize(w) for w in temp if w not in stop_words))

print(l1)



# vocabulary size

vocab_size = len(tokenizer.word_index) + 1
df_train['text'] = pd.DataFrame(l1)



df_train.head()
train_fin = df_train[['text', 'target']]

test_fin = df_test[['text']]
X = train_fin['text']

y = train_fin['target']




max_features = 3000

maxlen = 250



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train_fin.text.values)



# Pad the data 

X = tokenizer.texts_to_sequences(train_fin.text.values)

X = pad_sequences(X, maxlen=maxlen)



X_final = tokenizer.texts_to_sequences(test_fin.text.values)

X_final =  pad_sequences(X_final, maxlen=maxlen)



X_train, X_test, Y_train, Y_test = train_test_split(

    X, 

    train_fin.target, 

    test_size = 0.33, 

    random_state = 42

)

    

model = Sequential()

model.add(Embedding(max_features, 128, input_length=maxlen))

model.add(Bidirectional(LSTM(units = 64, recurrent_dropout=0.2)))

model.add(Dropout(0.5))

model.add(Dense(units = 32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units = 16, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(

    X_train, 

    Y_train,

    batch_size = 64,

    epochs = 10,

    verbose = 1,

    validation_data=(X_test, Y_test),

)
score, acc = model.evaluate(X_test, Y_test, batch_size = 32)

y_pred_train = model.predict_classes(X_train)

y_pred_test = model.predict_classes(X_test)



print("loss: ", score, "\nacc: ",acc)

print("\nTeste\n", classification_report(Y_test, y_pred_test))

print("\nTreino\n", classification_report(Y_train, y_pred_train))
#model = model.fit(data_train.text, data_train.target)

y_preds = model.predict_classes(X_final)
submission.target = y_preds

submission.to_csv("submission.csv", index=False)
submission.head(50).T
submission.target.value_counts().plot.bar()