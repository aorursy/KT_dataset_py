import numpy as np
import pandas as pd
data = pd.read_csv("../input/fake-news-data/train.csv")
data.head()
# checking for Null values
data.isnull().sum()
# Dropping Null values
data = data.dropna()
data.isnull().sum()
x= data.drop('label',axis=1)
y = data['label']
temp = x.copy()
temp.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
stopwords= stopwords.words('english')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
corpus = []
for i in range(0,len(temp)):
    text = re.sub('[^a-zA-Z]', ' ', temp['title'][i])
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if not word in stopwords]
    text = ' '.join(text)
    corpus.append(text)
corpus    
voc_size=5000

from tensorflow.keras.preprocessing.text import one_hot

one_hot_repr =[one_hot(words,voc_size) for words in corpus]
one_hot_repr
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(one_hot_repr,padding='post',maxlen =20)
padded
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dropout, GlobalAveragePooling1D, Flatten, Dense
embed_dim = 40
model = Sequential([
    Embedding(voc_size,embed_dim,input_length=20),
    Bidirectional(LSTM(100)),
    #Flatten(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1,activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics =['accuracy'])

model.summary()
x = np.array(padded)
y = np.array(y)
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y,test_size=0.3,random_state=0)
history = model.fit(trainX,trainY, epochs =10, validation_data=(testX,testY),batch_size=64)
pred = model.predict_classes(testX)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testY,pred)
import seaborn as sns

sns.heatmap(cm,annot=True,fmt='g')
