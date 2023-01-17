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
Real = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
Real.head()
Fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
Fake.head()
Real.isnull().sum()
Fake.isnull().sum()
#Fake = Fake.drop
Fake.info()
Real.info()
Real['isfake']=1
Fake['isfake']=0

Fake.head()
Real.head()
data =pd.concat([Real,Fake], ignore_index=True)
data
data.drop(['date'], axis =1, inplace= True)
data
data['news'] = data['title'] +' '+data['text']
data.head()
data['news'][0]
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >3 and token not in stop_words:
            result.append(token)
    return result
data['clean'] = data['news'].apply(preprocess)
data['clean'][0]
list_words = []
for i in data.clean:
    for j in i:
        list_words.append(j)
print (format(len(list_words)))

total_unique_words = len(list(set(list_words)))
total_unique_words
data['clean_joined'] = data['clean'].apply(lambda x: " ".join(x))
data['clean_joined'][0]
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,8))
sns.countplot(y = 'subject', data = data)
plt.figure(figsize=(8,8))
sns.countplot(y = 'isfake', data = data)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
maxlen = -1
for news in data.clean_joined:
    tokens = nltk.word_tokenize(news) #converts text to tokens (words)
    if (maxlen < len(tokens)):
        maxlen = len(tokens)
print (maxlen)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['clean_joined'], data['isfake'], test_size = 0.2, random_state = 42)
x_train[0]
import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words= total_unique_words)
tokenizer.fit_on_texts(x_train) #It creates vocabulary index ("word_index") based on word frequency
train_sequences = tokenizer.texts_to_sequences(x_train) # Replace each word in text with corresponding integer value from "word_index"
test_sequences = tokenizer.texts_to_sequences(x_test)
test_sequences[1]
len(train_sequences)
len(test_sequences)
print ("The encoding for news\n", data['clean_joined'][0], "\n is\n :", train_sequences[0])
pad_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen = 4405, padding = 'post', truncating= 'post')
pad_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=4405, padding = 'post', truncating= 'post')
pad_train[0]
pad_test[0]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
model = tf.keras.models.Sequential([

tf.keras.layers.Embedding(total_unique_words, output_dim = 128), #Embedding Layer

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)), #Bi-directional LSTM

#Dense layer
tf.keras.layers.Dense(128, activation = 'relu'),
tf.keras.layers.Dense(1, activation = 'sigmoid')])# binary classification (0\1)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['acc'])

model.summary()
y_train = np.asarray(y_train)
model.fit(pad_train, y_train, batch_size= 64, validation_split = 0.2, epochs= 2)
resluts = model.predict(pad_test)
prediction = []
for i in range (len(resluts)):
    if resluts[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print (accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot = True)
