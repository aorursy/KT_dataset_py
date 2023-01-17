# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/training.1600000.processed.noemoticon.csv",encoding='latin-1')
header= ['target','id','date','flag','user','text']
data.set_axis(header,axis=1,inplace=True)
data_ready=data.drop(['id','date','flag','user'],axis=1)
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from nltk.stem import PorterStemmer
from keras.callbacks import EarlyStopping, ModelCheckpoint
punct = list(string.punctuation)
stopword_list = stopwords.words('english') + punct + ['rt','via', '...']
stemmer= PorterStemmer()
# define a function for data cleaning / preprocessing
def sentense_to_words(raw_review):
    text = raw_review.lower()      
    tokens = TweetTokenizer().tokenize(text=text)
    clean_tokens= [stemmer.stem(tok) for tok in tokens if tok not in stopword_list and not tok.isdigit() and not tok.startswith('@')and not tok.startswith('#')and not tok.startswith('http')] 
    return( " ".join(clean_tokens))
# test the function for one tweet
tweet=sentense_to_words( data_ready['text'][1599973])
print(data_ready['text'][1599973], len(tweet) )
print(tweet)
# processing all tweets
corpus=[]
sent_len_list=[]
for i in range(0,len(data_ready)):
    corp= sentense_to_words(data_ready['text'][i])
    sent_len_list.append(len(corp))
    corpus.append(corp)
max_len=50
max_features=20000
# creating vectorized corpus and padding
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=max_len)
# relabel the sentiments 4 as 1
label= data_ready['target'].values
new_label=list(map(lambda x:x if x!= 4 else 1,label))
Y=to_categorical(new_label)
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=42)
classifier= Sequential()
classifier.add(Embedding(max_features,100,mask_zero=True))
classifier.add(LSTM(200,dropout=0.3,recurrent_dropout=0.3,return_sequences=False))
classifier.add(Dense(2, activation='softmax'))
classifier.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
classifier.summary()
callback = [EarlyStopping(monitor='val_loss', patience=2),ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
classifier.fit(X_train, y_train,batch_size=100,epochs=5,callbacks=callback ,validation_data=(X_test, y_test))