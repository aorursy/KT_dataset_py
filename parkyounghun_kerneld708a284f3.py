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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

import re

from nltk.stem import WordNetLemmatizer
def cleansing(text):

    pattern='([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'

    text=re.sub(pattern=pattern, repl='',string=text)

    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '<[^>]*>'

    text = re.sub(pattern=pattern, repl='', string=text)

    pattern = '[^\w\s]'         # 특수기호제거

    text = re.sub(pattern=pattern, repl='', string=text)

    return text

def cleansing_corpus(corpus):

    for i,text in enumerate(corpus):

        corpus[i]=cleansing(text)

    return corpus



def word_lemma(text):

    lem_text=[]

    lem=WordNetLemmatizer()

    lem_text=[lem.lemmatize(word) for word in text.split()]

    lem_text=' '.join(lem_text)

    return lem_text



def lemma_corpus(corpus):

    for i,text in enumerate(corpus):

        corpus[i]=word_lemma(text)

    return corpus



def preprocessing_corpus(corpus):

    corpus=cleansing_corpus(corpus)

    corpus=lemma_corpus(corpus)

    

    return corpus



        

tweet_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tweet_df_cleaned=preprocessing_corpus(tweet_df['text'])

X_train,X_test,y_train,y_test=train_test_split(tweet_df_cleaned,tweet_df.target,test_size=0.2)

cnt_vect=CountVectorizer()

cnt_vect.fit(X_train)

X_train_cnt_vect=cnt_vect.transform(X_train)



X_test_cnt_vect=cnt_vect.transform(X_test)



lr_clf=LogisticRegression()

lr_clf.fit(X_train_cnt_vect,y_train)

pred=lr_clf.predict(X_test_cnt_vect)

print(confusion_matrix(y_test,pred))

print('accuracy:{0:.3f}'.format(accuracy_score(y_test,pred)))

print('f1_score:',np.round(f1_score(y_test,pred),3))
tfid_vect=TfidfVectorizer()

tfid_vect.fit(X_train)

X_train_tfid=tfid_vect.transform(X_train)

X_test_tfidf=tfid_vect.transform(X_test)



lr_clf=LogisticRegression()

lr_clf.fit(X_train_tfid,y_train)

pred=lr_clf.predict(X_test_tfidf)

print(confusion_matrix(y_test,pred))

print('accuaracy:',np.round(accuracy_score(y_test,pred),3))

print('f1:',np.round(f1_score(y_test,pred),3))
disaster=tweet_df[tweet_df['target']==1]['text']

fake=tweet_df[tweet_df['target']==0]['text']



disaster=preprocessing_corpus(disaster)

fake=preprocessing_corpus(fake)



cnt_vect=CountVectorizer(stop_words='english',max_df=0.8,min_df=2)

disaster_vect=cnt_vect.fit_transform(disaster)

disaster_words_cnt=cnt_vect.vocabulary_.items()

words_freq_disaster=sorted(disaster_words_cnt, key = lambda x: x[1], reverse=True)

top10_disaster=words_freq_disaster[:10]



fake_vect=cnt_vect.fit_transform(fake)

fake_words_cnt=cnt_vect.vocabulary_.items()

words_freq_fake=sorted(fake_words_cnt, key = lambda x: x[1], reverse=True)

top10_fake=words_freq_fake[:10]
top10_disaster
top10_fake
tfid_vect=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_df=0.8,min_df=2)

tfid_vect.fit(X_train)

X_train_tfid=tfid_vect.transform(X_train)

X_test_tfidf=tfid_vect.transform(X_test)



lr_clf=LogisticRegression(C=1)

lr_clf.fit(X_train_tfid,y_train)

pred=lr_clf.predict(X_test_tfidf)

print(confusion_matrix(y_test,pred))

print('accuaracy:',np.round(accuracy_score(y_test,pred),3))

print('f1:',np.round(f1_score(y_test,pred),3))
test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

tfid_vect=TfidfVectorizer()

tfid_vect.fit(tweet_df['text'])

X_train_vect=tfid_vect.transform(tweet_df['text'])

X_test_vect=tfid_vect.transform(test_df['text'])



lr_clf=LogisticRegression()

lr_clf.fit(X_train_vect,tweet_df['target'])

pred=lr_clf.predict(X_test_vect)
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission['target']=pred
submission.to_csv('sample_submission.csv', index=False)