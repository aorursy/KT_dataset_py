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
data=pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
data.head()
import re
def process(x):
    processed_tweet = re.sub(r'\W', ' ', str(x))
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    processed_tweet = processed_tweet.lower()
    return processed_tweet
data.review=data.review.apply(process)
import nltk
from nltk.stem import PorterStemmer,LancasterStemmer
stemming =PorterStemmer()
def identify_tokens(row):
    tokens = nltk.word_tokenize(row)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
def stem_list(row):
    stemmed_list = [stemming.stem(word) for word in row]
    return (stemmed_list)
def rejoin_words(row):
    joined_words = ( " ".join(row))
    return joined_words
data.review=data.review.apply(identify_tokens)
data.review=data.review.apply(stem_list)
data.review=data.review.apply(rejoin_words)
data.head()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tk=Tokenizer(num_words=10000,oov_token='oov<>')
tk.fit_on_texts(data.review)
text_dict=tk.word_index
text_token=tk.texts_to_sequences(data.review)
pad=pad_sequences(text_token,maxlen=100,padding='post')
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
labels=la.fit_transform(data.review)
labels=np.array(labels)
pad=np.array(pad)
num_class=len(np.unique(labels))
from sklearn.model_selection import train_test_split
xr,xt,yr,yt=train_test_split(pad,labels,test_size=0.1)
import keras
yr=keras.utils.to_categorical(yr,num_class)
yt=keras.utils.to_categorical(yt,num_class)
from keras.layers import Embedding,GlobalAvgPool1D,Dense,Dropout
from keras.models import Sequential
model=Sequential()
model.add(Embedding(10000,10,input_length=100))
model.add(GlobalAvgPool1D())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_class,activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history=model.fit(xr,yr,epochs=2,batch_size=128)
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.legend(['Training Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.legend(['Training Loss'])
plt.show()