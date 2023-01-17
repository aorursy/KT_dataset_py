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

# Reference Code : https://www.kaggle.com/hassanamin/lstm-sentiment-analysis-data-imbalance-keras/edit
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from sklearn.utils import resample

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix,classification_report

import re

# Load Data

data = pd.read_csv('../input/Sentiment.csv')

# Keeping only the neccessary columns

data = data[['text','sentiment']]

data = data[data.sentiment != "Neutral"]
# Inspect data

data.head()
# Pre-Processing

data['text'] = data['text'].apply(lambda x: x.lower())

# removing special chars

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

data['text'] = data['text'].str.replace('rt','')

data.head()
max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

# Padding

X = pad_sequences(X)



Y = pd.get_dummies(data['sentiment']).values

# Train/Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 128

NoOfEpochs = 30

model.fit(X_train, Y_train, epochs = NoOfEpochs, batch_size=batch_size, verbose = 1)
Y_pred = model.predict_classes(X_test,batch_size = batch_size)

df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})

df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))

print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))

print(classification_report(df_test.true, df_test.pred))
# Separate majority and minority classes

data_majority = data[data['sentiment'] == 'Negative']

data_minority = data[data['sentiment'] == 'Positive']



bias = data_minority.shape[0]/data_majority.shape[0]

# lets split train/test data first then 

train = pd.concat([data_majority.sample(frac=0.8,random_state=200),

         data_minority.sample(frac=0.8,random_state=200)])

test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8,random_state=200).index),

        data_minority.drop(data_minority.sample(frac=0.8,random_state=200).index)])



train = shuffle(train)

test = shuffle(test)
print('positive data in training:',(train.sentiment == 'Positive').sum())

print('negative data in training:',(train.sentiment == 'Negative').sum())

print('positive data in test:',(test.sentiment == 'Positive').sum())

print('negative data in test:',(test.sentiment == 'Negative').sum())

# Separate majority and minority classes in training data for upsampling 

data_majority = train[train['sentiment'] == 'Negative']

data_minority = train[train['sentiment'] == 'Positive']



print("majority class before upsample:",data_majority.shape)

print("minority class before upsample:",data_minority.shape)



# Upsample minority class

data_minority_upsampled = resample(data_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples= data_majority.shape[0],    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

data_upsampled = pd.concat([data_majority, data_minority_upsampled])

 

# Display new class counts

print("After upsampling\n",data_upsampled.sentiment.value_counts(),sep = "")



max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['text'].values) # training with whole data



X_train = tokenizer.texts_to_sequences(data_upsampled['text'].values)

X_train = pad_sequences(X_train,maxlen=29)

Y_train = pd.get_dummies(data_upsampled['sentiment']).values

print('x_train shape:',X_train.shape)



X_test = tokenizer.texts_to_sequences(test['text'].values)

X_test = pad_sequences(X_test,maxlen=29)

Y_test = pd.get_dummies(test['sentiment']).values

print("x_test shape", X_test.shape)
# model

embed_dim = 128

lstm_out = 192



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.4, recurrent_dropout=0.4))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 128

# also adding weights

class_weights = {0: 1 ,

                1: 1.6/bias }

model.fit(X_train, Y_train, epochs = 30, batch_size=batch_size, verbose = 1,

          class_weight=class_weights)
Y_pred = model.predict_classes(X_test,batch_size = batch_size)

df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})

df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))

print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))

print(classification_report(df_test.true, df_test.pred))
# running model to few more epochs

model.fit(X_train, Y_train, epochs = 30, batch_size=batch_size, verbose = 1,

          class_weight=class_weights)

Y_pred = model.predict_classes(X_test,batch_size = batch_size)

df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})

df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))

print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))

print(classification_report(df_test.true, df_test.pred))