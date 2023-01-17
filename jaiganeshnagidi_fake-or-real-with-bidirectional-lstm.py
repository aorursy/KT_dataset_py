#importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#reading all the files
train=pd.read_csv("/kaggle/input/fake-news/train.csv")
test=pd.read_csv("/kaggle/input/fake-news/test.csv")
sub=pd.read_csv("/kaggle/input/fake-news/submit.csv")
train.head(5)
train.shape
train.isna().sum()
train=train.dropna() #dropping null values
train.reset_index(inplace=True) #resetting the index
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
sns.countplot(x='label',data=train)
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional

x=train.drop('label',axis=1)
y=train.label
y.value_counts()
#vocab size 
vocab_size = 5000
 
#one hot representation

messages = x.copy()

messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
#data set preprocessing 

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    result = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    result = result.lower()
    result = result.split()
    result = [ps.stem(word) for word in result if not word in stopwords.words('english')]
    result = ' '.join(result)
    corpus.append(result)
#Convert it into one hot vectors

onehot_rep = [one_hot(words,vocab_size) for words in corpus]
onehot_rep

#set a maximum length for sentences
smax_length=20
#embedded representation
embedd = pad_sequences(onehot_rep,padding='pre',maxlen=smax_length)
dims=40
bi_model=Sequential()
bi_model.add(Embedding(vocab_size,dims,input_length=smax_length))
bi_model.add(Bidirectional(LSTM(100))) #lstm with 100 neurons
bi_model.add(Dense(1,activation='sigmoid'))
bi_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(bi_model.summary())

len(embedd)
y.shape
x_final = np.array(embedd)
y_final = np.array(y)
x_final.shape,y_final.shape
x_final
y_final
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.3)
bi_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=64)
#create a model
from tensorflow.keras.layers import Dropout
dims=40
bi_model=Sequential()
bi_model.add(Embedding(vocab_size,dims,input_length=smax_length))
bi_model.add(Dropout(0.3))
bi_model.add(Bidirectional(LSTM(100))) #lstm with 100 neurons
bi_model.add(Dropout(0.3))
bi_model.add(Dense(1,activation='sigmoid'))
bi_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(bi_model.summary())
bi_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=64)
y_pred = bi_model.predict_classes(x_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_pred,y_test)
print("accuracy score :",accuracy_score(y_pred,y_test))
print("classification report :",classification_report(y_pred,y_test))
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(cm,annot=True)
plt.title("Confusion Matrix")
plt.show()