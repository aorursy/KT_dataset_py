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
data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print(data.shape)

print(test.shape)
data.head()
test['target']=sub['target']
data=pd.concat([data,test])
data.shape
data.isnull().sum()
data['target'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(x="target", data=data)
import string

import re

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
data['text']=data['text'].apply(lambda x:clean_text(x))
import nltk

from nltk.corpus import stopwords



tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

data['text'] = data['text'].apply(lambda x: tokenizer.tokenize(x))
data.head()
def remove_stopword(text):

    word=[w for w in text if w not in stopwords.words('english')]

    return word

data['text']=data['text'].apply(lambda x: remove_stopword(x))
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text





data['text'] = data['text'].apply(lambda x : combine_text(x))
data.shape
X=data.drop(['keyword','location','target'],axis=1)

y=data['target']
X.head()
text=data['text']
'''

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000,ngram_range=(1,3))

X = cv.fit_transform(data['text']).toarray()

'''


## TFidf Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_v=TfidfVectorizer(min_df=2, max_df=1.0, ngram_range=(1, 2))

X=tfidf_v.fit_transform(data['text']).toarray()

X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
X_test.shape
TF_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())
TF_df
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(C=4.9,max_iter=100000)

lr.fit(X_train,y_train)

pred = lr.predict(X_test)
score = accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)
#count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
#count_df.head()
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

classifier=MultinomialNB(alpha=0.59)
classifier.fit(X_train, y_train)

pred1 = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred1)

print("accuracy:   %0.3f" % score)
submission = pd.DataFrame({'id':test['id'],'target':pred1})



#Visualize the first 5 rows

submission.head()

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)