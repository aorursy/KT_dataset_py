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
X = pd.read_csv("/kaggle/input/fake-news/train.csv")

X.head()
X.shape
X.info()
X = X.dropna()

X.shape
x_train = []

x_train = X['title']

x_train = np.asarray(x_train)

print(x_train.shape)

type(x_train)

x_train
import regex as re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()
X_train=[]



for i in range(len(x_train)):

    rev = re.sub('[^a-zA-Z]',' ',x_train[i])

    rev = rev.lower()

    rev = rev.split()

    rev = [ps.stem(word) for word in rev if word not in stopwords.words('english')]

    review = " ".join(rev) 

    X_train.append(review)



X_train[0]
len(X_train)
vocab_size = 5000



from keras.preprocessing.text import one_hot



X_train_hot = [one_hot(i,vocab_size) for i in X_train]

X_train_hot[0]
[i.insert(0,1) for i in X_train_hot]
max_rev_len = 25



from keras.preprocessing import sequence



train_set = sequence.pad_sequences(X_train_hot,maxlen=max_rev_len)

train_set[10]
train_labels = X['label']

train_labels
print(train_set.shape)

print(len(train_labels))
print(type(train_set))

print(type(train_labels))



train_labels = np.asarray(train_labels)

print(type(train_labels))
from keras.models import Sequential

from keras.layers import LSTM,Dense

from keras.layers.embeddings import Embedding
model = Sequential()



model.add(Embedding(vocab_size,

                   32,

                   input_length=max_rev_len))

model.add(LSTM(100))

model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



model.summary()
model.fit(train_set,train_labels,

         validation_split=0.25,

         verbose=2,

         epochs=10,

         batch_size=256)
test = pd.read_csv("/kaggle/input/fake-news/test.csv")

test
test_data= []

test_data= test['title']

test_data= np.asarray(test_data)

print(test_data.shape)

type(test_data)

test_data
test_data[0]
Test=[]



for i in range(len(test_data)):

    rev = re.sub('[^a-zA-Z]',' ',str(test_data[i]))

    rev = rev.lower()

    rev = rev.split()

    rev = [ps.stem(word) for word in rev if word not in stopwords.words('english')]

    review = " ".join(rev) 

    Test.append(review)



Test[0]
type(Test)
Test_hot = [one_hot(i,vocab_size) for i in Test]



[i.insert(0,1) for i in Test_hot]



test_set = sequence.pad_sequences(Test_hot,maxlen=max_rev_len)

test_set[0]
type(test_set)

test_set.shape
preds = model.predict(test_set)

preds
preds = np.round(preds).astype("int8")
preds.shape
submission_data = pd.read_csv('/kaggle/input/fake-news/submit.csv')
submission_data['label']=preds

submission_data
submission_data.to_csv('Submission.csv',index=False)