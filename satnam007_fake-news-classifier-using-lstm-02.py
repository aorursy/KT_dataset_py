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
df=pd.read_csv('../input/fake-news/train.csv')
df.head()
###Drop Nan Values
df=df.dropna()
## Get the Independent Features
X=df.drop('label',axis=1)

## Get the Dependent features
y=df['label']

print(X.shape)
print(y.shape)
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
### Vocabulary size
voc_size=5000
messages=X.copy()

print(messages.head(1))
print(messages['title'][1])
#as we have droped the nan values so we have deleted the index sequence
messages.reset_index(inplace=True)
import nltk
import re  #Regular expressions
from nltk.corpus import stopwords
nltk.download('stopwords')
### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    #print('\n', messages['title'][i])
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])# ^ means except #remove all things except [^a-zA-Z]
    #print('\n', review)
    review = review.lower() # make all words in lower case
    #print('\n', review)
    review = review.split() # split all words
    #print('\n', review)
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # looking each word and remove stopWords
    #print('\n', review)
    review = ' '.join(review) # join all words with space
    #print('\n', review)
    corpus.append(review) # append the whole line one by one
    #break
corpus[4]
#converting the all words into oneHot
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
print(onehot_repr[0])
print('\n',len(onehot_repr))
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
len(embedded_docs),y.shape
import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape,y_final.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(500))
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
### Finally Training again
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64)
### Finally Training again
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
# test_data = pd.read_csv('../input/fake-news/test.csv')
# test_data.head()
# test_data.isnull().sum()
# test_data.dropna(axis=0, inplace=True)
# test_data.reset_index(inplace=True)
# test_data['title'].head(15)
# #test data preprocessing
# corpus = []
# for i in range(0, len(test_data)):
#     review = re.sub('[^a-zA-Z]', ' ', test_data['title'][i])
#     review = review.lower()
#     review = review.split()
    
#     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#     review = ' '.join(review)
#     corpus.append(review)
# corpus[0]
# onehot_repr=[one_hot(words,voc_size)for words in corpus] 
# onehot_repr[0]
# sent_length=20
# test=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
# print(embedded_docs)
# y_pred=model.predict_classes(test)
from sklearn.metrics import confusion_matrix
y_pred

