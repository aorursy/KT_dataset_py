import pandas as pd
df1=pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')

df1.head()
df1.shape
df=df1[['description','requirements','fraudulent']]

df.head()
###Drop Nan Values

df=df.dropna()
df.shape
## Get the Independent Features



X=df.drop('fraudulent',axis=1)
## Get the Dependent features

y=df['fraudulent']
y.value_counts()
import tensorflow as tf

tf.__version__
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Dropout
voc_size=5000
message = X.copy()
message['description'][1]
message.reset_index(inplace=True)
import nltk

import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0, len(message)):

    review = re.sub('[^a-zA-Z]', ' ', message['description'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
corpus[1]
onehot_repr=[one_hot(words,voc_size)for words in corpus] 

onehot_repr[1]
sent_length=40

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs)
embedded_docs[0]
## Creating model

embedding_vector_features=50

model1=Sequential()

model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model1.add(Bidirectional(LSTM(100))) ##Just add bidirectional!!, except it would just behave as normal LSTM Model

model1.add(Dropout(0.3))

model1.add(Dense(1,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model1.summary())
len(embedded_docs),y.shape
import numpy as np

X_final=np.array(embedded_docs)

y_final=np.array(y)
X_final[1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=32)
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=12,batch_size=64)
y_pred=model1.predict_classes(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))