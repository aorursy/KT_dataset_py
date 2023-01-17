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
train = pd.read_csv('../input/train.tsv',delimiter='\t')
train.head()
test = pd.read_csv('../input/test.tsv',delimiter='\t')
test.head()
train.shape
train.dtypes
train.describe()
train.info()
train.columns
train.Sentiment.nunique()
train.Sentiment.unique()
train.isnull().sum()
train.PhraseId.duplicated().sum()
train.PhraseId.nunique()
train.SentenceId.nunique()
train.SentenceId.duplicated().sum()
train.SentenceId.value_counts()[:10]
train[train.SentenceId==1][:5]
train[train.SentenceId==128][:5]
train.Sentiment.value_counts()
train.Phrase[0]
train[['Sentiment','Phrase']].sort_values('Sentiment',ascending=True)[:10]
train["Phrase"] = train["Phrase"].astype(str,copy=True) 

type(train.Phrase)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
cv = CountVectorizer()
x = train.Phrase
x
y = train.Sentiment.values
y

x_cv  = cv.fit_transform(x)
x_cv
type(x_cv)
x_cv.toarray()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_cv,y)
lr.score(x_cv,y)
x_test = test.Phrase
x_test
x_test_cv = cv.transform(x_test)
x_test_cv
x_test_cv.toarray()
preds = lr.predict(x_test_cv)
preds
type(preds)
lr.score(x_cv,y)
from sklearn.naive_bayes import MultinomialNB,GaussianNB
mnb = MultinomialNB()
gnb = GaussianNB()
mnb.fit(x_cv,y)
mnb.score(x_cv,y)
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,RandomForestClassifier

dtc = DecisionTreeClassifier()
etc = ExtraTreesClassifier()
abc = AdaBoostClassifier()
rfc = RandomForestClassifier()
c_list = [dtc,etc,abc,rfc]
scores = []
for i in c_list:
    i.fit(x_cv,y)
    score = i.score(x_cv,y)
    scores.append(score)
print(scores)
for i,j in zip(c_list,scores):
    print('{} score is {}'.format(str(i),j))      
print('the best classifier score  is {}'.format(max(scores)))