# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
fake_data = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true_data = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
fake_data.head()
true_data.head()
true_data['class'] = 1

fake_data['class'] = 0
# Merging both the datasets

data = pd.concat([true_data,fake_data])
data.head()
data['class'].value_counts().plot(kind = 'bar', color = ['g','b'])

plt.xlabel('Class')

plt.ylabel('No. of particulars')
# Checking for missing values

data.isnull().sum()
data['subject'].value_counts()
# Merging the title and text column

data['text'] = data['title'] + " " + data['text']



# Dropping the title, subject and date columns

data.drop(columns = ['title','date','subject'], inplace = True)
data.head()
split = np.random.rand(len(data)) < 0.7

train = data[split]

test = data[~split]
# Checking whether target classes are balanced

# If they are balanced in the train set, then we do not need to check the test set as well

train['class'].value_counts()
train['class'].value_counts().plot(kind = 'bar', color = ['r','b'])
train.head()
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import re

import string
# Removing whitespaces and single word characters from the text

train['text'] = train.text.apply(lambda x: re.sub(r'[^\w\s]', '',x))

test['text'] = test.text.apply(lambda x: re.sub(r'[^\w\s]', '',x))
train['text'] = [re.sub(r'\b\w{1,3}\b', '', c) for c in train['text']]

test['text'] = [re.sub(r'\b\w{1,3}\b', '', c) for c in test['text']]
# Removing numeric characters

train['text'] = [re.sub('\d','',n) for n in train['text']]

test['text'] = [re.sub('\d','',n) for n in test['text']]
# Converting all text to lower case characters

train['text'] = [t.lower() for t in train['text']]

test['text'] = [t.lower() for t in test['text']]
train['text'].dropna(inplace = True)

test['text'].dropna(inplace = True)
#import nltk

#regex_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

#train['text'] = train['text'].map(regex_tokenizer.tokenize())

#test['text'] = test['text'].map(regex_tokenizer.tokenize())
import nltk

from nltk.tokenize import word_tokenize



# Word Tokenization

train['text'] = [word_tokenize(i) for i in train['text']]

test['text'] = [word_tokenize(i) for i in test['text']]
from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))

train['text'] = [[i for i in j if not i in stop_words] for j in train['text']]

test['text'] = [[i for i in j if not i in stop_words] for j in test['text']]
#train.drop(columns = ['text'], inplace = True)

#test.drop(columns = ['text'], inplace = True)
from tensorflow.keras.preprocessing.text import Tokenizer
train['text'] = train['text'].apply(lambda x : ' '.join(x))

test['text'] = test['text'].apply(lambda x : ' '.join(x))
X = train.text

test_X = test.text

train_labels = train['class'].values

test_labels = test['class'].values
from collections import Counter



# Finding the number of unique word in the corpus

def word_counter(text):

    count = Counter()

    for i in text.values:

        for word in i.split():

            count[word] += 1

    return count
counter = word_counter(X)

words = len(counter)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train['text'])
train_X = tokenizer.texts_to_sequences(train['text'])
# Numeric tokens for the first 25 words of the the 5th news entry in our train set

train_X[5][:25]
test_X = tokenizer.texts_to_sequences(test['text'])
test_labels = test['class'].values
word_index = tokenizer.word_index

for word, num in word_index.items():

    print(f"{word} -> {num}")

    if num == 10:

        break        
nos = np.array([len(x) for x in train_X])

len(nos[nos  < 500])
len(counter)
from keras.preprocessing.sequence import pad_sequences



#Lets keep all news with upto 500 words, add padding to news with less than 500 words

maxlen = 500 



#Making all news of size max length defined above

train_X = pad_sequences(train_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)
# All sequences have been made of length 500

# if a news had more than 500 words they have been truncated to 500

# if a news had less than 500 that sequence has been padded with 0

len(train_X[20])
from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam

import tensorflow as tf



def leaky_relu(z, name = None):

    return tf.maximum(0.01*z,z, name = name)



model = Sequential()



model.add(Embedding(words+1,32,input_length = 500)) # embedding layer

model.add(LSTM(64)) # RNN layer

#model.add(Dense(units = 32 , activation = leaky_relu)) # Dense layer with leaky_relu activation

model.add(Dense(1, activation = 'sigmoid'))



optimizer = Adam(learning_rate = 3e-4)



model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()
#Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X,train_labels) 
model.fit(train_X, train_labels, validation_split=0.2, epochs=10)
predictions = model.predict_classes(test_X)
from sklearn.metrics import accuracy_score, classification_report

accuracy_score(test_labels,predictions)
print(classification_report(test_labels,predictions))