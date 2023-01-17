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
df_yelp=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/yelp_labelled.txt',sep='\t',header=None)
df_yelp.head()
df_imdb=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/imdb_labelled.txt',sep='\t',header=None)
df_amzn=pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/amazon_cells_labelled.txt',sep='\t',header=None)
df_yelp.shape, df_amzn.shape , df_imdb.shape
col_names=['review','sentiment']

df_yelp.columns=col_names

df_imdb.columns=col_names

df_amzn.columns=col_names
df_yelp.head()
df_yelp.loc[20]['review']
data=df_yelp.append([df_amzn,df_imdb],ignore_index=True)
data.head()
data.shape
data['sentiment'].value_counts()
data.isnull().sum()
import string

punc=string.punctuation
punc
import spacy

nlp=spacy.load('en_core_web_sm')
x='hello! as its WorlD'

doc=nlp(x)

for token in doc:

    print(token.lemma_)
def lemmatize(x):

    doc=nlp(x)

    tokens=[]

    for token in doc:

        if token.lemma_ != '-PRON-':

            temp=token.lemma_.lower().strip()

        else:

            temp=token.lower_

        tokens.append(temp)

    return tokens
lemmatize(x)
from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)
import string

punc=string.punctuation
def stop_word_and_punc(x):

    tokens=[]

    for token in x:

        if token not in STOP_WORDS and token not in punc:

            tokens.append(token)

    return tokens
def data_cleaning(x):

    tokens=lemmatize(x)

    return stop_word_and_punc(tokens)
x="Hello my name is shubham and this is good learning drinking runs learned"

data_cleaning(x)
data.head()
text=" ".join(data['review'])
text
from spacy import displacy

nlp=spacy.load('en_core_web_sm')
doc=nlp(text)
displacy.render(doc,style='pos')
displacy.render(doc,style='ent')
import matplotlib.pyplot as plt
data['word_count']=data['review'].apply(lambda x:len(x.split()))
data.head()
def get_char_count(x):

    count=0

    for word in x.split():

        count+=len(word)

    return count
data['char_count']=data['review'].apply(lambda x:get_char_count(x))
data.head()
plt.hist(data[data['sentiment']==0]['word_count'],bins=200)

plt.hist(data[data['sentiment']==1]['word_count'],bins=200)

plt.xlim([0,60])

plt.show()
plt.hist(data[data['sentiment']==0]['char_count'],bins=200)

plt.hist(data[data['sentiment']==1]['char_count'],bins=200)

plt.xlim([0,300])

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC,SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.pipeline import Pipeline
tfidf=TfidfVectorizer(tokenizer=data_cleaning)

classifier=SVC()
X=data['review']

y=data['sentiment']
X_train,X_test,y_train,y_test= train_test_split(X,y,shuffle=True,random_state=0,test_size=0.2)
X_train.shape, X_test.shape 
clf=Pipeline([('tfidf',tfidf),('clf',classifier)])
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

report=classification_report(y_test,y_pred)

print(report)
cm=confusion_matrix(y_test,y_pred)

print(cm)
import spacy

nlp=spacy.load('en_core_web_sm')
x="hello world apple mango"
doc=nlp(x)
for token in doc:

    print(token.text, token.has_vector , token.vector.shape)
def get_vector(x):

    doc=nlp(x)

    return doc.vector.reshape(-1,1)
data['vector']=data['review'].apply(lambda x:get_vector(x))
data.head()
data.loc[3]['vector'].shape
import tensorflow as tf
X=np.concatenate(data['vector'].to_numpy(),axis=1)

X=np.transpose(X)

y=(data['sentiment']>1).astype(int)
X.shape , y.shape
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,LSTM

from tensorflow.keras.models import Sequential
model=Sequential([

    Dense(128,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(64,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(2,activation='sigmoid')

])
import tensorflow as tf

y_oh=tf.keras.utils.to_categorical(y,num_classes=2)
X_train,X_test,y_train,y_test=train_test_split(X,y_oh,random_state=2,test_size=0.2)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=[X_test,y_test])
from sklearn.metrics import confusion_matrix

y_pred=model.predict(X_test)

y_pred=np.argmax(y_pred,axis=1)

y_pred.shape , y_test.shape

cm=confusion_matrix(y_test,y_pred)

cm