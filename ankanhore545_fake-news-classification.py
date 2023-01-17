import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df=pd.read_csv('../input/fake-news-data/train.csv')
df.head()
#Dropping the NAN values
df=df.dropna()
#Getting the Independant features
X= df.drop('label', axis= 1)

#Getting the dependant features
y= df['label']
X.shape

y.shape
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #due to classification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
voc_size=10000
m= X.copy()
m.reset_index(inplace= True)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
ps= PorterStemmer()

for i in range(len(m)):
    text= re.sub(r'\[^a-zA-Z]', ' ', m['title'][i])
    text= text.lower()
    text= re.sub(r'\d+', ' ', text)
    text= re.sub(r'\s+', ' ', text)
    text= [ps.stem(word)for word in text if not word in stopwords.words('english')]
    text= ''.join(text)
    corpus.append(text)
onehot= [one_hot(words, voc_size) for words in corpus]
onehot
sent_length= 20
embed_docs= pad_sequences(onehot, padding= 'pre', maxlen= sent_length)
embed_docs
len(embed_docs)
#Creating the model

vector_features=40
model= Sequential()
model.add(Embedding(voc_size, vector_features, input_length=sent_length))
model.add(Dropout(0.3))#adding the dropout layer
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))#classification problem
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
len(embed_docs), y.shape
import numpy as np
X_final=np.array(embed_docs)
y_final=np.array(y)
X_final.shape, y_final.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_final, y_final, test_size=0.33, random_state=42)
#Finally training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
y_pred= model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)# for understaing the better classification and other accuracy parameters
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)