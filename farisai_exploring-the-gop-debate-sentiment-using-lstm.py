# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Sentiment.csv')
df.head()
# Exploring the number of columns
df.info()
#lets see how many subjects do we have here
df['subject_matter'].unique()
#Using seaborn to plot a visually pleasant charts
import seaborn as se
import matplotlib as mpl
import matplotlib.pyplot as plt

#Here we are just controlling the size of the grid and the orientation of the labels in the x-axis
plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.countplot(x="subject_matter", data=df, palette="Greens_d");

plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.barplot(x="subject_matter", y="sentiment_confidence", hue="sentiment", data=df);
df['tweet_location'].unique()
#Lets look at that in barplot
plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.countplot(x="tweet_location", data=df);
#We extract the two columns text and sentiment from the dataframe
data = df[['text','sentiment']]

data.head()
#This code borrowed from Peter Nagy "LSTM Sentiment Analysis | Keras"

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)
print(data[ data['sentiment'] == 'Neutral'].size)


for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
#Used the same model architecture used by Peter Nagy "LSTM Sentiment Analysis | Keras"

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 32
model.fit(X_train, Y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)
#Retrain again in 90 epochs
model.fit(X_train, Y_train, epochs = 90, batch_size=batch_size, verbose = 2)
embed_dim = 280
lstm_out = 210

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)
model.fit(X_train, Y_train, epochs = 90, batch_size=batch_size, verbose = 2)