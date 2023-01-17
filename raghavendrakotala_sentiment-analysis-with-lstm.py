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
df_train = pd.read_csv('../input/train.tsv',sep='\t')
df_test = pd.read_csv('../input/test.tsv',sep='\t')
#first describe the data
df_train.shape,df_test.shape
#top 5 rows
df_train.head()
df_test.head()
len(list(df_test.Phrase.values)+list(df_train.Phrase.values))
len(list(df_test.Phrase)+list(df_train.Phrase))
full_text = list(df_test.Phrase)+list(df_train.Phrase)
len(full_text)
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
df_train['clean_review'] = clean_review(df_train.Phrase)
df_test['clean_review'] = clean_review(df_test.Phrase)
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
train_text = df_train.clean_review.values
test_text = df_test.clean_review.values
y = to_categorical(df_train.Sentiment.values)
print(train_text.shape,test_text.shape,y.shape)
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)
all_words=' '.join(X_train_text)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
num_unique_word
len_x_train = []
for text in X_train_text:
    words = word_tokenize(text)
    len_x_train.append(len(words))
MAX_REVIEW_LEN=np.max(len_x_train)
#max features to be considered while tokening the training data
max_features = num_unique_word
#max length of each review
max_words = MAX_REVIEW_LEN
#as we can't pass entire dataset into training once we divide each epoches into no of iterations so the size of data in each iteratios is called batch size
batch_size = 128
#word embedding vecotr lenght for each word in high dimensional space
embedding_vecor_length = 250
#number of time entire data passed forward and backward through entire network 
epochs = 10
#no of classes in the output layer
num_classes=y.shape[1]
num_classes
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))

X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)
#we will make each review to same length as max_review length so that every review as of same length, for less length padd with zeros
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape,y.shape)
#instantiate keras model with sequential constructor
model=Sequential()
#embedding layer is used to represent words in the meaningful vectors in a dimensional space, so it takes total number words,size of vector,
model.add(Embedding(max_features,250,mask_zero=True))
#LSTM layer with 128 neurons, dropout for configuring the input dropout and recurrent_dropout for configuring the recurrent dropout
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
#return_sequences=True argument,
#What this does is ensure that the LSTM cell returns all of the outputs from the unrolled LSTM cell through time. If this argument is left out, the LSTM cell will simply provide the output of the LSTM cell from the last time step
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
#since it's classification problem we use dense layer with no of classes.
model.add(Dense(num_classes,activation='softmax'))
#finally compiling model, loss = 'categorical_crossentropy' because of many classes, adam optimizer because its effective “all round” with adaptive stepping, metrics is ‘categorical_accuracy’ --  which can let us see how the accuracy is improving during training
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
#summary of model
model.summary()
%%time

history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=5, batch_size=batch_size, verbose=1)
#lets predict on the test data
y_pred=model.predict_classes(X_test)
y_pred
sub=pd.read_csv('../input/sampleSubmission.csv')
sub.head()
sub.Sentiment=y_pred
sub.head()
sub.to_csv('sub1.csv',index=False)
