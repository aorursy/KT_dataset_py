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

import nltk

import re

from nltk.corpus import stopwords
dataset_fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

dataset_fake["label"]=1

dataset_true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

dataset_true["label"]=0
dataset_fake.head(3)
dataset_true.head(3)
dataset = pd.concat((dataset_fake,dataset_true), axis=0)

dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.head()
X = dataset.title

y = dataset.label
X.head()
y.head()
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(len(X)):

    news = re.sub( "[^a-zA-Z]"," ",X[i])

    news = news.lower()

    news = news.split()

    sub_corpus = [ps.stem(word) for word in news if not word in stopwords.words("english")]

    sub_corpus = " ".join(sub_corpus)

    corpus.append(sub_corpus)

corpus[:5]
from tensorflow.keras.preprocessing.text import one_hot

input_len = 1000

oh = [one_hot(word, input_len) for word in corpus]

oh[:5]
from tensorflow.keras.preprocessing.sequence import pad_sequences

ohp = pad_sequences(oh, padding = "post")

sent_len = len(ohp[0])

ohp[:5]
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential
output_len = 50

model = Sequential()

model.add(Embedding(input_len, output_len, input_length=sent_len))

# model.add(Dropout(0.3))

# model.add(flatten()) Not required in Deep learning

model.add(LSTM(100))

# model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print(model.summary())
X_final = np.array(ohp)

y_final = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size =0.2, random_state=20)
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10, batch_size=64)

y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)