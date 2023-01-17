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
test=pd.read_csv(r"/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip",sep="\t")

train=pd.read_csv(r"/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip",sep="\t")

sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
#lets look at the test data

test.head()
#lets look at the test data

train.head(100)
# lets look at the shape of the train data

train.shape
# lets look at the shape of the test data

test.shape
train.loc[train['SentenceId']==3]
train.loc[train['SentenceId']==2]
# since we have different values in  sentenceId,check the total no of unique sentenceId 

print("For train data ",train['SentenceId'].nunique()) 

print("For test data ",test['SentenceId'].nunique()) 


pd.DataFrame(train.groupby('SentenceId')['Phrase'].count()).head(10)
## Returning average count of phrases per sentence, per Dataset

int(train.groupby('SentenceId')['Phrase'].count().mean())
int(test.groupby('SentenceId')['Phrase'].count().mean())
#Returning average word length of phrases

print("train ",int(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))

print("test",int(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
train_count=train['Sentiment'].value_counts() 
#gets the unique value count of an object

train_labels=train['Sentiment'].value_counts().index
import seaborn as sns

import matplotlib.pyplot as plt

fig,ax = plt.subplots(1, 1, dpi = 100, figsize = (7, 5))

g=sns.barplot(train_labels,train_count)

ax.set_xlabel("target")

ax.set_ylabel("count")
import nltk
tokenizer = nltk.tokenize.TweetTokenizer()
# import tfidf vectoriser

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

vectorizer.fit(full_text)  #learns both train and test data vocabulary

train_vectorized = vectorizer.transform(train['Phrase'])

test_vectorized = vectorizer.transform(test['Phrase'])
train_vectorized.shape
y = train['Sentiment']
test_vectorized.shape
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

ovr = OneVsRestClassifier(logreg)
ovr.fit(train_vectorized, y)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=10)
print("Mean of 10 cv :",np.mean(scores) * 100)

print( "standard deviation",np.std(scores) * 100)
y_test=ovr.predict(test_vectorized)
sub.Sentiment=y_test
sub.head()
sub.to_csv('submission.csv',index=False)