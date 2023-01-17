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
from nltk.corpus import stopwords
from nltk.util import ngrams
import re
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import one_hot
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
sample=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
sample.head()
sns.barplot(train.target.value_counts().index,train.target.value_counts())
plt.gca().set_ylabel('samples')
mess=train.text
ps=PorterStemmer()
corpus=[]
for i in range(0,len(mess)):
    r=re.sub('https?://\S+|www\.\S+',' ',mess[i])
    r=re.sub(r'<.*?>',' ',r)
    r=re.sub('[^a-zA-Z]',' ',r)
    r=r.lower()
    r=r.split()
    r=[ps.stem(word) for word in r if not word in stopwords.words('english')]
    r=' '.join(r)
    corpus.append(r)
corpus
len_of_word=[]
for i in range(0,len(train.text)):
    a=len(train.text[i])
    len_of_word.append(a)
sent_len=max(len_of_word)+3
sent_len
vocab_size=8000
ohe=[one_hot(word,vocab_size) for word in corpus]
ohe
embed_doc=pad_sequences(ohe,padding='pre',maxlen=sent_len)
print(embed_doc)
model=Sequential()
model.add(Embedding(vocab_size,100,input_length=sent_len))
model.add(LSTM(100))#dropout=0.2
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
xfinal=np.array(embed_doc)
yfinal=np.array(train.target)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xfinal,yfinal,test_size=0.15,random_state=143)
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=64)
ypred=model.predict(xtest)

y_pred=np.round(ypred).astype(int).reshape(1142)
y_pred
mess_final=test.text
corpus_final=[]
for i in range(0,len(mess_final)):
    res=re.sub('https?://\S+|www\.\S+',' ',mess_final[i])
    res=re.sub(r'<.*?>',' ',res)
    res=re.sub('[^a-zA-Z]',' ',res)
    res=res.lower()
    corpus_final.append(res)

len(corpus_final)
vocab_size=8000
ohe_final=[one_hot(word,vocab_size) for word in corpus_final]
embed_doc_final=pad_sequences(ohe_final,padding='pre',maxlen=160)
embed_doc_final.shape
ypred_final=model.predict(embed_doc_final)
pred_final=np.round(ypred_final).astype(int).reshape(3263)
pred_final
sub=pd.DataFrame({'id':test['id'].values.tolist(),'target':pred_final})
sub.to_csv('submission.csv',index=False)
