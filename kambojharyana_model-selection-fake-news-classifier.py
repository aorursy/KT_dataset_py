#importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#libraries used 

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
# raw data is the data which is output from previous notebook (gathering fake news dataset)
data = pd.read_csv('../input/my-data/data.csv',index_col = 0)
print(data.shape)
data = data.reset_index(drop = True)
data.head()
#encoding the label col
data['label'] = np.where(data['label'] == 'Fake',0,1)
# 0 - fake , 1 - true
data.head()
# lets see the value counts of the classes 
data['label'].value_counts() / len(data)
## delete it later
data['text'][0]
text = list(data['text'])
stop_words = set(stopwords.words('english'))

ps=PorterStemmer()
corpus=[]

from tqdm import tqdm 

for t in tqdm(text):
    result = re.sub('[^a-zA-Z]',' ',t)
    result = result.lower()
    result = result.split()
    result = [ps.stem(word) for word in result if not word in stop_words]
    result = ' '.join(result)
    corpus.append(result)

vocab_size = 10000
onehot_rep = [one_hot(words,vocab_size) for words in corpus]
onehot_rep[:2] #observing first two elements
#set a maximum length for sentences
smax_length= 20
#embedded representation
embedd = pad_sequences(onehot_rep,padding='pre',maxlen=smax_length)
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
#creating x and y 
y = np.array(data['label'])
X = np.array(embedd)
print(y.shape)
print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
bi_model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)
test_preds = bi_model.predict_classes(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy is ',accuracy_score(y_test,test_preds))
print('')
print('Confusion matrix is ')
confusion_matrix(y_test,test_preds)
bi_model.save('my_model.h5')
