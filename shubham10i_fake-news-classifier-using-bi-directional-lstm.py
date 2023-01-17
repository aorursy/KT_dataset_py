import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM,Dense,Bidirectional

import nltk
import re
from nltk.corpus import stopwords
train = pd.read_csv('../input/fake-news/train.csv')
test = pd.read_csv('../input/fake-news/test.csv')
train.head()
test.head()
train.isna().sum()
train = train.dropna()
train.isna().sum()
test.isna().sum()
test = test.dropna()
test.isna().sum()
X = train.drop('label',axis=1)
X
y = train['label']
y
plt.style.use('fivethirtyeight')
sns.countplot(data=train,x='label')
X.shape
voc_size = 5000
messages = X.copy()
messages['title'][1]
messages.reset_index(inplace=True)
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('^[a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
test_data = test.copy()
test_data.reset_index(inplace=True)
test_data['title'][1]

test_corpus = []

for i in range(0,len(test_data)):
    review = re.sub('^[a-zA-Z]',' ',test_data['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    test_corpus.append(review)
test_corpus
corpus
onehot_test_rep = [one_hot(words,voc_size) for words in test_corpus]
onehot_test_rep
onehot_rep = [one_hot(words,voc_size) for words in corpus]
onehot_rep
sent_length = 25
embedded_test_docs = pad_sequences(onehot_test_rep,padding='pre',maxlen=sent_length)
print(embedded_test_docs)
sent_length = 25
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]
embedded_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedded_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
len(embedded_docs),y.shape
X_test_final = np.array(embedded_test_docs)
X_final = np.array(embedded_docs)
y_final = np.array(y)
X_final.shape,y_final.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,y_final,test_size=0.3,random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=64,epochs=20)
y_pred = model.predict_classes(X_test_final)
y_pred
y_pred = np.array(y_pred)
y_pred
