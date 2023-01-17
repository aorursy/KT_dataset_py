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
df=pd.read_csv('/kaggle/input/fake-news/train.csv')
df.shape
#Drop the NAN values
df=df.dropna()
#drop the label from the dataset
X=df.drop('label',axis=1)
X.head()
y=df['label']
y

X.shape
# y.shape
import numpy as np
a=np.array(y)


import tensorflow as tf

tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
messages=X.copy()
messages['title'][0]
messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
wordnet=WordNetLemmatizer ()
c=[]
for i in range(0,len(messages)):
    p=re.sub('[^A-Za-z]',' ',messages['title'][i])
    p=p.lower()
    p=p.split()
    p=[wordnet.lemmatize(word) for word in p if not word in stopwords.words('english')]
    p=' '.join(p)
    c.append(p)
c[0]
voc_size=10000
onehot=[one_hot(words,voc_size)for words in c]
onehot
total_len=25
embedding=pad_sequences(onehot,padding='pre',maxlen=total_len)
embedding
# len(embedding)
embedding[0]
Embedding_features=50
model=Sequential()
model.add(Embedding(voc_size,Embedding_features,input_length=total_len))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
len(embedding),y.shape
rea=pd.read_csv('/kaggle/input/fake-news/submit.csv')

rea['label']
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=42)
X_train=np.array('embedding')

dff=pd.read_csv('/kaggle/input/fake-news/test.csv')
dff.shape
dff.id
# dff['id']
dff=dff.fillna(' ')
mess=dff.copy()

mess['title'][0]
len(mess)
d=[]
for i in range(0,len(mess)):
    q=re.sub('[^A-Za-z]',' ',mess['title'][i])
    q=q.lower()
    q=q.split()
    q=[wordnet.lemmatize(word) for word in q if not word in stopwords.words('english')]
    q=' '.join(q)
    d.append(q)
d[0]
onhot=[one_hot(words,voc_size)for words in d]
onhot
embeding=pad_sequences(onhot,padding='pre',maxlen=total_len)
len(embeding)
import numpy as np
X_final=np.array(embeding)
y_final=rea['label']
import numpy as np
x_train=np.array(embedding)
y_train=np.array(y)
# X_final.shape,y_final.shape
# x_train.shape,y_train.shape
# y.isnull().sum()
# embedding
# x_train.shape
y_train.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=42)
model.fit(x_train,a,validation_data=(X_test,y_test),epochs=50,batch_size=64)
y_pred=model.predict_classes(X_final)
# y_pred
result = y_pred.flatten()
result
submission = pd.DataFrame({'id':dff.id, 'label':result})
submission.to_csv('submission.csv', index=False)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_final,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_final,y_pred)

