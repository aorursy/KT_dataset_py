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

df=pd.read_csv('../input/fake-news/train.csv')
df.head()
df=df.dropna()
x=df.drop('label',axis=1)
y=df['label']
y.value_counts()
x.shape
y.shape
df.head(10)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Dropout
voc_size=10000
mes=x.copy()

mes['title'][1]
mes.reset_index(inplace=True)
import nltk

import re

from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]



for i in range(0,len(mes)):

    review=re.sub('[^a-zA-Z]',' ',mes['title'][i])

    review=review.lower()

    review=review.split()

    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]

    review=' '.join(review)

    corpus.append(review)
corpus
one_hot_rep=[one_hot(word,voc_size) for word in corpus]

one_hot_rep
sent_len=20

emb_doc=pad_sequences(one_hot_rep,padding='pre',maxlen=sent_len)

print(emb_doc)
emb_doc[0]
embedding_vector_features=40

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))

model.add(Bidirectional(LSTM(100)))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
len(emb_doc),y.shape
import numpy as np

x_final=np.array(emb_doc)

y_final=np.array(y)
x_final.shape,y_final.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=42)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=64)
y_pred=model.predict_classes(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
y_test
y=np.resize(y_test,5200)

s=pd.read_csv('../input/fake-news/submit.csv')
submission=pd.DataFrame.from_dict({

    'id':s.id,

    'label':y

})
submission.to_csv("submit.csv",index=False)