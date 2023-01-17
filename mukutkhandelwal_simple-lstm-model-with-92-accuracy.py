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
import re

import pandas as pd

import nltk

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

nltk.download('stopwords')

nltk.download('wordnet')

train_data = pd.read_csv("../input/fake-news/train.csv")

test_data = pd.read_csv("../input/fake-news/test.csv")
train_data.head(3)

test_data.head(3)

train_data.shape

test_data.shape

test_data.isnull().sum()

train_data.isnull().sum()

train_data.fillna("missing",inplace = True)

test_data.fillna("missing",inplace = True)
x = train_data.drop(columns="label")

y = train_data['label']
x_copy = x.copy()

ws = WordNetLemmatizer()

new_list = []

# new_list1 = []

for i in range(len(x_copy)):

    titles = re.sub('[^a-zA-z]'," ",x_copy['title'][i])

    titles = titles.lower()

    titles = titles.split()

    titles = [ws.lemmatize(word) for word in titles if word not in stopwords.words("english")]

    titles = " ".join(titles)

    new_list.append(titles)
vocab_size = 10000

one_hot_title = [one_hot(i,vocab_size) for i in new_list]
new_list[0]
one_hot_title[0]
res = len(max(new_list, key = len))

res
sent_len = 356

embed_title = pad_sequences(one_hot_title,maxlen = sent_len,padding = 'pre')
embed_title.shape
model = Sequential()

model.add(Embedding(input_dim=vocab_size,output_dim = 40,input_length=356))

model.add(LSTM(150))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

plot_model(model)
model.fit(embed_title,y,epochs = 7,validation_split=0.2)
model.evaluate(embed_title,y)
test_data[:3]
test_list = []

for i in range(len(test_data)):

    title = re.sub("[^a-zA-Z]"," ",test_data['title'][i])

    title = title.lower()

    title = title.split()

    title = [ws.lemmatize(word) for word in title if word not in stopwords.words('english')]

    title = " ".join(title)

    test_list.append(title)

    
one_hot_test = [one_hot(i,vocab_size) for i in test_list]

embed_test = pad_sequences(one_hot_test,maxlen = 356,padding = 'pre')
ypred = model.predict(embed_test)

pred = (ypred>0.5).astype('int')
mysubmit = pd.DataFrame(data = pred,columns = ['label'])

mysubmit['id'] = test_data['id']
mysubmit.head()
