import pandas as pd
    
#read Train dataset
df=pd.read_csv('/kaggle/input/fake-news/train.csv')
df.head()
# drop nan values from train sets
df=df.dropna()
X=df.drop('label',axis=1)
X.shape
y=df['label']
y.shape
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
voc_size=5000
message=X.copy()
message.head(100)
message.reset_index(inplace=True)
message.head()
import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(message)):
    #print(i)
    
    if i % 5000==0:
        print(i)
    review=re.sub('[^a-zA-Z]',' ',message['text'][i])
    review=review.lower()
    words=review.split()
    
    reviews=[ps.stem(word) for word in words if word not in stopwords.words('english')]
    review=' '.join(reviews)
    
    corpus.append(review)
    
one_hotRepr=[one_hot(word,voc_size)for word in corpus]
one_hotRepr[0]
# padding is added to make input of equal length

sent_length=20
embedded_docs=pad_sequences(one_hotRepr,padding='pre',maxlen=sent_length)
embedded_docs
#embedded_docs=pad_sequences(one_hotRepr,padding='post',maxlen=sent_legth)
#

embedding_vectors_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vectors_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.compile('adam','mse')
model.summary()
import numpy as np

X_final=np.array(embedded_docs)
y_final=np.array(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.3,random_state=50)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
testdf=pd.read_csv('/kaggle/input/fake-news/test.csv')
testdf
# Drop Nan Values
X = testdf.fillna('is the')
    
messages = X.copy()

messages.reset_index(inplace = True)
messages['title'][9]
testcorpus=[]
for i in range(0,len(testdf)):
    #print(i)
    
    if i % 2000==0:
        print(i)
    
    review=re.sub('[^a-zA-Z]',' ',messages['text'][i])
    review=review.lower()
    words=review.split()
    
    reviews=[ps.stem(word) for word in words if word not in stopwords.words('english')]
    review=' '.join(reviews)
    
    testcorpus.append(review)
    
testone_hotRepr=[one_hot(word,voc_size)for word in testcorpus]
sent_length=20
testembedded_docs=pad_sequences(testone_hotRepr,padding='pre',maxlen=sent_length)
testembedded_docs
#embedded_docs=pad_sequences(one_hotRepr,padding='post',maxlen=sent_legth)

test=np.array(testembedded_docs)
predict=model.predict_classes(test)
(len(predict),len(test))

df_sub = pd.DataFrame()
df_sub['id'] = testdf['id']
df_sub['label'] = predict
df_sub.to_csv('My_submit.csv', index=False)
df_sub.head()

