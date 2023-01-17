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
#data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#string
import string
import re


#text processing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from textblob import TextBlob
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.info()
test.info()
#dropping the null valued columns 
train.drop(['keyword','location'],inplace=True,axis=1)
test.drop(['keyword','location'],inplace=True,axis=1)
#count statistics
def words_count(df):
    df['word_count']=df['text'].apply(lambda x: len([x for x in x.split()]))
    print(df[['word_count','text']])
words_count(train)
words_count(test)
train['word_count'].describe()
#lowercasing
def lowercasing(df):
    df['cleaned']=df['text'].apply(lambda x:' '.join([x.lower() for x in x.split()]))
    
#removing url links
def remove_URL(df):
    df['cleaned']=df['cleaned'].apply(lambda x:' '.join([x for x in x.split() if x[:3]!='http']))

#remove punctuation
def remove_punctuation(df):
    df['cleaned']=df['cleaned'].str.replace('[^\w\s]','')

#removing stopwords
def remove_stopwords(df):
    stop=stopwords.words('english')
    df['cleaned']=df['cleaned'].apply(lambda x:' '.join([x for x in x.split() if x not in stop]))

#lemmatization
def lemmatization(df):
    lemm=WordNetLemmatizer()
    df['cleaned']=df['cleaned'].apply(lambda x:' '.join([lemm.lemmatize(x) for x in x.split()]))
    return (df[['text','cleaned']].head())


    

lowercasing(train)
remove_URL(train)
remove_punctuation(train)
remove_stopwords(train)
lemmatization(train)

lowercasing(test)
remove_URL(test)
remove_punctuation(test)
remove_stopwords(test)
lemmatization(test)


X_train=train['cleaned'][:-2284]
X_val=train['cleaned'][-2284:]
y_train=train['target'][:-2284]
y_val=train['target'][-2284:]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000)
cv.fit(train['cleaned'])
X_train_cv=cv.transform(X_train)
X_val_cv=cv.transform(X_val)
test_cv=cv.transform(test['cleaned'])
test_cv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


clf=LogisticRegression()
clf.fit(X_train_cv,y_train)
predictions=clf.predict(X_val_cv)
clf.score(X_train_cv,y_train)
from sklearn.metrics import accuracy_score,classification_report,precision_score
print('accuracy score: ',accuracy_score(predictions,y_val))
print('claasification report: \n',classification_report(y_val,predictions))
print('confusion matrix \n',confusion_matrix(y_val,predictions))
df=pd.concat([train['cleaned'],test['cleaned']])
len(df)
from nltk.tokenize import word_tokenize
from tqdm import tqdm

corpus=[]
def create_corpus(df):
    for tweet in tqdm(df):
        words=[word for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus
corpus=create_corpus(df)
embedding_dict={}
with open('../input/glove6b100dtxt/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
        
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX,padding='post',truncating='post')


word_index=tokenizer_obj.word_index
word_index
print('Total no of unique words are: ',len(word_index))
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i>num_words:
        continue
        
    embedding_vec=embedding_dict.get(word)
    
    if embedding_vec is not None:
        embedding_matrix[i]=embedding_vec
    
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
model=Sequential()
embedding=layers.Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),input_length=MAX,trainable=False)
model.add(embedding)
model.add(layers.SpatialDropout1D(0.2))
model.add(layers.LSTM(64,dropout=0.2,recurrent_dropout=0.2))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy',metrics=['accuracy'])
train_df=tweet_pad[:train.shape[0]]
test_df=tweet_pad[train.shape[0]:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_df,train['target'],test_size=0.2,random_state=123)
X_train.shape
X_test.shape
hist=model.fit(X_train,y_train,epochs=15,verbose=1,validation_data=(X_test,y_test))
