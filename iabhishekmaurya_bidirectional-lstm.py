import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

import re
import pickle
import numpy as np
import pandas as pd

import nltk
import re
from nltk.corpus import stopwords
columns = ["sentiment", "ids", "date", "flag", "user", "text"]
enc = "ISO-8859-1"
data = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',
                      encoding=enc , names=columns)
data.head()
data.shape
data.isnull().sum()
data.info()
data['sentiment'].describe()
data['sentiment'].value_counts()
X = data['text']
y = data['sentiment']
v_size = 50000
from nltk.stem import WordNetLemmatizer
def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
     
        # Replace @USERNAME to ' '.
        tweet = re.sub(userPattern,' ', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText
import time
t = time.time()
corpus = preprocess(X)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')
corpus
med(data['text'].apply(lambda x: len(x.split(" "))))
t = time.time()
onehot_repr=[one_hot(words,40000)for words in corpus] 
print(f'Time Taken: {round(time.time()-t)} seconds')
onehot_repr
sent_length=110
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]
embedding_vector_features=220
model1=Sequential()
model1.add(Embedding(110,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())
len(embedded_docs),y.shape
import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape,y_final.shape
y.value_counts()
y = y.replace(4,1)
y.value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=25)
### Finally Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)