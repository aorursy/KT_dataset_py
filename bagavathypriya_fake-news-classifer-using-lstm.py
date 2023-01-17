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
df = pd.read_csv("../input/fake-news/train.csv")
df.head()
df.describe()
df.info()
#Dropping the NaN values
df = df.dropna()
df.info()
#Independent variables

x = df.drop('label',axis=1)
x.head()

#depedent variable
y=df['label']
y.head()
x.shape
y.shape
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
#Vocabulary
vocab = 5000
msg = x.copy()
msg.reset_index(inplace=True)
msg.head()
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
# Data preprocessing

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(msg)):
    print(i)
    #Substituting letters other than a-z with blank spaces
    review = re.sub('[^a-zA-Z]', ' ', msg['title'][i]) 
    review = review.lower()
    review = review.split()
    
    #Taking only the words from review which are not the stopwords
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus
onehot_rep = [one_hot(words,vocab) for words in corpus]
onehot_rep
sen_len = 20
embed = pad_sequences(onehot_rep,padding='pre',maxlen=sen_len)
embed
embed[0]
embedding_vector_features = 40

model = Sequential()

model.add(Embedding(vocab,embedding_vector_features,input_length=sen_len))

model.add(LSTM(100))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
x_final = np.array(embed)
y_final = np.array(y)
x_final.shape
y_final.shape
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_final,y_final,test_size=0.33,random_state=42)
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=64)
ypred = model.predict_classes(xtest)
from sklearn.metrics import confusion_matrix
confusion_matrix(ypred,ytest)
from sklearn.metrics import accuracy_score
accuracy_score(ypred,ytest)*100
#Adding some dropout layers in order to reduce the overfitting problem

from tensorflow.keras.layers import Dropout

embedding_vector_features = 40

model = Sequential()

model.add(Embedding(vocab,embedding_vector_features,input_length=sen_len))

model.add(Dropout(0.3))

model.add(LSTM(100))

model.add(Dropout(0.3))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=64)
ypred = model.predict_classes(xtest)

confusion_matrix(ypred,ytest)
accuracy_score(ypred,ytest)*100
