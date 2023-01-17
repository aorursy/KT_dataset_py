# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
import pandas as pd

sampleSubmission = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv")


train = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/train.tsv', sep="\t")

test = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/test.tsv', sep="\t")

sub = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv', sep="\t")
train['Sentiment'].unique()
# df = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/train.tsv",sep='\t')

# df
# from sklearn.feature_extraction.text import TfidfVectorizer

# import nltk.tokenize as tokenizer

# from nltk import word_tokenize
# from keras.preprocessing import sequence, text

# from keras.preprocessing.text import Tokenizer

# from keras.models import Sequential

# from keras.preprocessing.sequence import pad_sequences



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from nltk.tokenize import TweetTokenizer

import datetime

#import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score

#from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

pd.set_option('max_colwidth',400)


from nltk.tokenize import word_tokenize

from nltk import FreqDist

from nltk.stem import SnowballStemmer,WordNetLemmatizer

stemmer=SnowballStemmer('english')

lemma=WordNetLemmatizer()

from string import punctuation

import re
# df['cleaned']=df['Phrase'].str.lower()

# df['cleaned']
# vectorizer = TfidfVectorizer(ngram_range=(1,2),tokenizer=tokenizer.tokenize)

# X = vectorizer.fit_transform(df['cleaned'])


def clean_review(review_col):

    review_corpus=[]

    for i in range(0,len(review_col)):

        review=str(review_col[i])

        review=re.sub('[^a-zA-Z]',' ',review)

        

        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]

        review=' '.join(review)

        review_corpus.append(review)

    return review_corpus
train['clean_review']=clean_review(train.Phrase.values)

train.head()

#Upsampling for a larger train set

from sklearn.utils import resample

train_2 = train[train['Sentiment']==2]

train_1 = train[train['Sentiment']==1]

train_3 = train[train['Sentiment']==3]

train_4 = train[train['Sentiment']==4]

train_5 = train[train['Sentiment']==0]

train_2_sample = resample(train_2,replace=True,n_samples=75000,random_state=123)

train_1_sample = resample(train_1,replace=True,n_samples=75000,random_state=123)

train_3_sample = resample(train_3,replace=True,n_samples=75000,random_state=123)

train_4_sample = resample(train_4,replace=True,n_samples=75000,random_state=123)

train_5_sample = resample(train_5,replace=True,n_samples=75000,random_state=123)



df_upsampled = pd.concat([train_2, train_1_sample,train_3_sample,train_4_sample,train_5_sample])

df_upsampled.head()
# Apply to test data

test['clean_review']=clean_review(test.Phrase.values)

test.head()
text = ' '.join(df_upsampled.loc[df_upsampled.Sentiment == 4, 'Phrase'].values)

text_trigrams = [i for i in ngrams(text.split(), 3)]
text_bigrams = [i for i in ngrams(text.split(), 2)]
Counter(text_bigrams).most_common(5)
Counter(text_trigrams).most_common(5)
#Tokenize the data train and test

tokenizer = TweetTokenizer()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

full_text = list(df_upsampled['clean_review'].values) + list(test['clean_review'].values)

vectorizer.fit(full_text)

df_upsampled_vectorized = vectorizer.transform(df_upsampled['clean_review'])

test_vectorized = vectorizer.transform(test['clean_review'])

test1 = test['clean_review']



y = df_upsampled['Sentiment']
#Tokenize the data train and test with 3-gram

tokenizer = TweetTokenizer()

vectorizer3 = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)

# full_text = list(df_upsampled['clean_review'].values) + list(test['clean_review'].values)

vectorizer3.fit(full_text)

df_upsampled_vectorized3 = vectorizer3.transform(df_upsampled['clean_review'])

test_vectorized3 = vectorizer3.transform(test['clean_review'])

# test1 = test['clean_review']



# y = df_upsampled['Sentiment']
logreg = LogisticRegression()

ovr = OneVsRestClassifier(logreg)
%%time

ovr.fit(df_upsampled_vectorized, y)
%%time

ovr3 = OneVsRestClassifier(logreg)

ovr3.fit(df_upsampled_vectorized3, y)
ovrpredictions3 = ovr3.predict(test_vectorized3)

ovrpredictions3
ovrresult3 = test

ovrresult3['Sentiment'] = ovrpredictions3

ovrresult3.drop(['clean_review','SentenceId','Phrase'],axis=1).to_csv('ovr3.csv',index=False)
# Cross validation



scores = cross_val_score(ovr, df_upsampled_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)

print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
#Linear SVC
svc = LinearSVC(dual=False)

svc.fit(df_upsampled_vectorized, y)

svcpred = svc.predict(test_vectorized)

svcresult = test

svcresult['Sentiment'] = svcpred

svcresult.drop(['clean_review','SentenceId','Phrase'],axis=1).to_csv('svc.csv',index=False)
#Cross validation with LinearSVC

# %%time

svc = LinearSVC(dual=False)

scores = cross_val_score(svc, df_upsampled_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)

print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
mnb =  MultinomialNB()

mnb.fit(df_upsampled_vectorized, y)

mnbpred = mnb.predict(test_vectorized)

mnbresult = test

mnbresult['Sentiment'] = mnbpred

mnbresult.drop(['clean_review','SentenceId','Phrase'],axis=1).to_csv('mnb1.csv',index=False)
#Cross validation with MultinomialNB



# %%time

model = MultinomialNB()

#model.fit(train_vectorized, y)

scores =  cross_val_score(model, df_upsampled_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)

print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
#convert sentiment to categories to use in LSTM

from keras.utils import to_categorical

X = df_upsampled['clean_review']

#test_set = test['clean review']

#Y = train['Sentiment']

Y = to_categorical(df_upsampled['Sentiment'].values)

print(Y)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=123)


print(X_train.shape,Y_train.shape)

print(X_val.shape,Y_val.shape)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
all_words=' '.join(X_train)

all_words=word_tokenize(all_words)

#print(all_words)

dist=FreqDist(all_words)



num_unique_word=len(dist)

num_unique_word

#X_train.head()
r_len=[]

for text in X_train:

    word=word_tokenize(text)

  #  print(text)

    l=len(word)

    r_len.append(l)

    

MAX_REVIEW_LEN=np.max(r_len)

MAX_REVIEW_LEN
#Setting features

max_features = num_unique_word

max_words = MAX_REVIEW_LEN

batch_size = 128

epochs = 3

num_classes=5
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, GRU, Embedding

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)



X_test = tokenizer.texts_to_sequences(test1)

#X_test
from keras.preprocessing import sequence,text

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences


from keras.preprocessing import sequence,text

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_val = sequence.pad_sequences(X_val, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)

#print(X_train.shape,X_val.shape)

X_test
from keras.preprocessing import sequence,text

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score

import matplotlib.pyplot as plt
model1=Sequential()

model1.add(Embedding(max_features,100,mask_zero=True))



model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))

model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

model1.add(Dense(num_classes,activation='softmax'))





model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model1.summary()


#%%time

model1.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)
pred1=model1.predict_classes(X_test,verbose=1)
sub.Sentiment=pred1

sub.to_csv('sub1.csv',index=False)

sub.head()
from keras.layers import Input, Dense, Embedding, Flatten

from keras.layers import SpatialDropout1D

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.models import Sequential
model2 = Sequential()



# Input / Embdedding

model2.add(Embedding(max_features, 150, input_length=max_words))



# CNN

model2.add(SpatialDropout1D(0.2))



model2.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))

model2.add(MaxPooling1D(pool_size=2))



model2.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))

model2.add(MaxPooling1D(pool_size=2))



model2.add(Flatten())



# Output layer

model2.add(Dense(5, activation='sigmoid'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)
pred2=model2.predict_classes(X_test,verbose=1)

sub.Sentiment=pred2

sub.to_csv('sub2.csv',index=False)

sub.head()
model3= Sequential()

model3.add(Embedding(max_features,100,input_length=max_words))

model3.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))

model3.add(MaxPooling1D(pool_size=2))

model3.add(Dropout(0.25))

model3.add(GRU(128,return_sequences=True))

model3.add(Dropout(0.3))

model3.add(Flatten())

model3.add(Dense(128,activation='relu'))

model3.add(Dropout(0.5))

model3.add(Dense(5,activation='softmax'))

model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model3.summary()
%%time

model3.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)
pred3=model3.predict_classes(X_test,verbose=1)

sub.Sentiment=pred3

sub.to_csv('sub3.csv',index=False)

sub.head()