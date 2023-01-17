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
trainData=pd.read_csv('/kaggle/input/fake-news/train.csv')

data=trainData

testData=pd.read_csv('/kaggle/input/fake-news/test.csv')

data.head()
data.shape
testData.head()
data['author'].nunique()
import seaborn as sns



sns.countplot('label',data=data)
data.isnull().sum()
testData.isnull().sum()
#Dropping Nan values

data=data.dropna()

data.shape

X_train=data.drop('label',axis=1)

X_train.head(1)
y_train=data['label']
y_train.shape
import tensorflow as tf

tf.__version__
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense
#Vocabulary Size

voc_size=5000
message=X_train.copy()

message.reset_index(inplace=True)



import nltk

import re

from nltk.corpus import stopwords
nltk.download('stopwords')
print(len(message))
#Preprocessing of Data



from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

#Creating a List

corpus=[]



for i in range(0,len(message)):

    print(i)

    review=re.sub('[^A-Za-z]',' ',message['title'][i])

    review=review.lower()

    review=review.split()

    

    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]

    review=' '.join(review)

    corpus.append(review)
corpus
onehotData=[one_hot(words,voc_size)for words in corpus]

onehotData
sent_length=20

embeddedDoc=pad_sequences(onehotData,padding='pre',maxlen=sent_length)

print(embeddedDoc)
embeddedDoc[0]
#Creating Model

embedding_vector_features=40

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model.add(LSTM(100))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
len(embeddedDoc),y_train.shape
X_final=np.array(embeddedDoc)

y_final=np.array(y_train)

X_final.shape,y_final.shape
from sklearn.model_selection import train_test_split

X,X_valid,y,y_valid=train_test_split(X_final,y_final,test_size=0.33,random_state=42)
#Final Training

model.fit(X,y,validation_data=(X_valid,y_valid),epochs=10,batch_size=64)