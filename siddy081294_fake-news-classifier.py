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
train_df = pd.read_csv('/kaggle/input/fake-news/train.csv')
train_df.head()
train_df.isnull().sum()
len(train_df)
train_df.dropna(inplace=True)
X = train_df.drop('label', axis=1)
y = train_df['label']
X.shape
y.shape
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM,Dense

from tensorflow.keras.preprocessing.text import one_hot
#Vocabulary -- dictionary of 5000 words to get all sentences

voc_size = 5000
message = X.copy()
message = message.reset_index() # to set the missing index for null values
import re

import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem.lancaster import LancasterStemmer
def preprocessing(column_data):

    ps = PorterStemmer()

    corpus = []

    for i in range(len(column_data)):

        print(i)

        review = re.sub('[^a-zA-Z]',' ', column_data[i])

        review = review.lower()

        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

        review = ' '.join(review)

        corpus.append(review)

    return corpus
preprocessed_title = preprocessing(message['title'])
preprocessed_title[30]
one_hot_repr_title = [one_hot(words,voc_size) for words in preprocessed_title]
one_hot_repr_title
sent_length = 20
embedded_title = pad_sequences(one_hot_repr_title, padding='pre', maxlen=sent_length)
embedded_title
embedded_test_title = pad_sequences(one_hot_repr_test_title, padding='pre', maxlen=sent_length)
# Creating Vectors
embedded_vec_features = 40
from tensorflow.keras.layers import Dropout

model= Sequential()

model.add(Embedding(voc_size,embedded_vec_features,input_length=sent_length))

model.add(Dropout(0.3))

model.add(LSTM(100))

model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
X_final = np.array(embedded_title)

y_final = np.array(y)
test = np.array(embedded_test_title)
X_final.shape, y_final.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,y_final,test_size=0.15,random_state=42)
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=15, batch_size=32)
test_df = pd.read_csv('/kaggle/input/fake-news/test.csv')
test_df.isnull().sum()
test_df.fillna('No data here', inplace=True)
test_df.head()
preprocessed_test_title = preprocessing(test_df['title'])
one_hot_repr_test_title = [one_hot(words,voc_size) for words in preprocessed_test_title]
embedded_test_title = pad_sequences(one_hot_repr_test_title, padding='pre', maxlen=sent_length)
testing_array = np.array(embedded_test_title)
y_pred = model.predict_classes(testing_array)
y_pred
y_pred.shape
label = [i[0] for i in y_pred]
id = test_df['id']
submission = pd.DataFrame({'id':id, 'label':label})
submission.to_csv('submit.csv', index=False)