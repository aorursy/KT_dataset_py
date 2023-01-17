#import basic module

import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns
RANDOM_SEED = 42
fake_news = pd. read_csv('../input/fake-and-real-news-dataset/Fake.csv')

true_news = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
print(fake_news.shape)

print(true_news.shape)
fake_news.head()
true_news.head()
words = []

words.append(list(fake_news['text'].apply(len)))

words.append(list(true_news['text'].apply(len)))

ax = sns.boxplot(data=words)

ax.set(xticklabels=['fake', 'true'])
words = []

words.append(list(fake_news['title'].apply(len)))

words.append(list(true_news['title'].apply(len)))

ax = sns.boxplot(data=words)

ax.set(xticklabels=['fake', 'true'])
import collections

def calc_unique_words(col: pd.Series):

    col = list(col)

    unique = set()

    for x in col:

        unique |= set(x.split())

    return len(unique)

unique_fake = calc_unique_words(fake_news['text'])

unique_true = calc_unique_words(true_news['text'])
print(unique_fake, unique_true)

# fake news have unique words
fake_news['subject'].value_counts()
true_news['subject'].value_counts()
fake_news['fake_flg'] = 1

true_news['fake_flg'] = 0
df = pd.concat([fake_news, true_news])
df.head()
df.tail()
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer 
import string

# eliminate puctuation

print(f'puncuations: {string.punctuation}')

nopunc = [c for c in df['title'] if c not in string.punctuation]
from tqdm.notebook import tnrange

corpus = []

for i in tnrange(len(df)):

    #elminate number, other signs

    title = re.sub('[^a-zA-Z]', ' ', nopunc[i]) 

    title = title.lower()

    title = title.split()

    

    #word stemming("likes"->"like")

    ps = PorterStemmer()

    title = [ps.stem(words) for words in title if not words in set(stopwords.words('english'))]



    title = ' '.join(title)

    corpus.append(title)
corpus[3]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(corpus, df['fake_flg'], test_size = 0.20, random_state = RANDOM_SEED)
from sklearn.pipeline import Pipeline 

from sklearn.feature_extraction.text import TfidfVectorizer

#vectorize text with tfidf(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)



tfidf = TfidfVectorizer()

tfidf.fit(X_train) #train should be done only with train data

X_train = tfidf.transform(X_train)

X_test = tfidf.transform(X_test)
#function for easy training and valuation

from sklearn.metrics import classification_report,roc_auc_score

def train_and_predict(clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred)

    print('auc: {:.5}'.format(auc_score))

    print(classification_report(y_test, y_pred))

    return clf, y_pred
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

train_and_predict(clf)
import lightgbm as lgb

clf = lgb.LGBMClassifier()

train_and_predict(clf)
#hyper parmerter seach 

for i in [50, 100, 200, 400, 1000]:

    print(f'num_leaves: {i}')

    clf = lgb.LGBMClassifier(num_leaves=i)

    train_and_predict(clf)