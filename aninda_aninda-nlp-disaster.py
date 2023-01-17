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
df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col='id')

df.head()
df['target'].value_counts()
print('The shape of the dataframe is:',df.shape)

#Number of null values

print('The number of null values in keyword column is:',df['keyword'].isnull().sum())

print('The number of null values in location column is:',df['location'].isnull().sum())
#data=df[['keyword','text','target']]

#data.dropna(axis=0,inplace=True)

data=df[['text','target']]

data.shape
data.head()
#lookup_dict=dict(data.groupby(['keyword']).mean())
#data.loc[:,'dummy_val']=data['keyword'].apply(lambda v: lookup_dict['target'][v])
print(data.head())

data.shape
import spacy

from spacy.lang.en.stop_words import STOP_WORDS

stopwords=STOP_WORDS

nlp = spacy.load('en_core_web_sm')

def preprocess(text):

    # Create Doc object

    doc = nlp(text, disable=['ner', 'parser'])

    # Generate lemmas

    lemmas = [token.lemma_ for token in doc]

    # Remove stopwords and non-alphabetic characters

    a_lemmas = [lemma for lemma in lemmas 

            if lemma not in stopwords]

    

    return ' '.join(a_lemmas)

  

# Apply preprocess to ted['transcript']

data['text'] = data['text'].apply(preprocess)
data.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

vectorizer=TfidfVectorizer()

X_train,X_test,y_train,y_test=train_test_split(data[['text']],data['target'],test_size=0.10,random_state=123,stratify=data['target'])

tfidf_matrix = vectorizer.fit_transform(X_train['text'])
X_train.head()
tfidf_matrix.shape
X_train.drop(['text'],axis=1,inplace=True)

X_train.shape
X_train.reset_index(inplace=True)

X_train.drop(['id'],axis=1,inplace=True)

X_train.head()
X_train=X_train.join(pd.DataFrame(tfidf_matrix.toarray()))

X_train.shape
X_train.head()
import xgboost as xgb

clf=xgb.XGBClassifier()

clf.fit(X_train,y_train)
tfidf_matrix_test = vectorizer.transform(X_test['text'])

tfidf_matrix_test.shape
X_test.drop(['text'],axis=1,inplace=True)

X_test.reset_index(inplace=True)

X_test.drop(['id'],axis=1,inplace=True)

X_test=X_test.join(pd.DataFrame(tfidf_matrix_test.toarray()))

X_test.shape
y_pred=clf.predict(X_test)

y_pred.shape
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col='id')

test.head()
print('The shape of the dataframe is:',test.shape)

#Number of null values

print('The number of null values in keyword column is:',test['keyword'].isnull().sum())

print('The number of null values in location column is:',test['location'].isnull().sum())
data_test=test[['text']]

#data_test.dropna(axis=0,inplace=True)

data_test.shape
#data_test['dummy_val']=data_test['keyword'].apply(lambda v: lookup_dict['target'][v])

#data_test.head()
data_test.reset_index(inplace=True)

data_test.drop(['id'],axis=1,inplace=True)

data_test.head()
data_test['text']=data_test['text'].apply(preprocess)
tfidf_matrix_pred = vectorizer.transform(data_test['text'])

tfidf_matrix_pred.shape
data_test.drop(['text'],axis=1,inplace=True)

data_test.head()
pd.DataFrame(tfidf_matrix_pred.toarray()).shape
data_test=pd.concat([data_test,pd.DataFrame(tfidf_matrix_pred.toarray())],axis=1)

data_test.shape
sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sub.shape
preds=clf.predict(data_test)

preds.shape
sub['target']=preds

sub.head()
sub.to_csv('submission.csv',index=False)