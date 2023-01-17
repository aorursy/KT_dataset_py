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
import matplotlib.pyplot as plt
import seaborn as sns
true=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
true.head()
fake=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
fake.head()
len(true),len(fake)
true['category']=1
fake['category']=0
df=pd.concat([true,fake])
len(df)
pd.crosstab(df['subject'],df['category'])
df.info()
import string
from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

#cleaning the data
from bs4 import BeautifulSoup

#removing html content
def strip_markup(text):
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text()


#removing square brackets from the text
import re

def remove_punc(text):
    text=re.sub("[^\w\s']",' ',text)
    return text
def remove_stopwords(text):
    text=' '.join([word.lower() for word in text.split() if word.lower() not in stop])
    return text
def clean_text(text):
    text=strip_markup(text)
    text=remove_punc(text)
    text=remove_stopwords(text)
    return text
df['text']=df['text'].apply(lambda x: clean_text(x))
df.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df['text'],df['category'],test_size=0.3,random_state=123)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features=10000
max_len=300
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
train_sequence=tokenizer.texts_to_sequences(x_train)
test_sequence=tokenizer.texts_to_sequences(x_test)
train_sequence=pad_sequences(train_sequence,maxlen=max_len,padding='post')
test_sequence=pad_sequences(test_sequence,maxlen=max_len,padding='post')

EMBEDDING_FILE='../input/glovetwitter27b100dtxt/glove.twitter.27B.100d.txt'

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

#change below line if computing normal stats is too slow
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words,100))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Embedding,LSTM,Dense

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
#Defining Neural Network
model = Sequential()
model.add(Embedding(max_features, output_dim=100, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
from tensorflow.keras import optimizers
model.compile(optimizer=optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_sequence, y_train, batch_size = 256 , validation_data = (test_sequence,y_test) , epochs =5 , callbacks = [learning_rate_reduction])





