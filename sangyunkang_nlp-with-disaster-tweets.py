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
# Basic packages -> Already declared

# import numpy as np 

# import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Natural Language Processing packages

import nltk

from nltk.corpus import stopwords

import string as s

import re

# Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

# ML packages

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV

from xgboost import XGBClassifier

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Evaluation pacakges

from sklearn.metrics import f1_score
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.shape
test.shape
train.dtypes
test.dtypes
train.info()
test.info()
train.head()
test.head()
train['target'].value_counts()
sns.countplot(x='target',data=train,palette='RdBu_r')
sns.barplot(y=train['keyword'].value_counts()[:10].index,x=train['keyword'].value_counts()[:10],orient='h')
train['location'].value_counts()
sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],orient='h')
pd.isnull(train).any()
pd.isnull(train).sum() # check the missing values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
import missingno as msno
msno.matrix(train)
msno.matrix(test)
# Check which variables have missing values



columns_with_missing_values = train.columns[train.isnull().any()]

train[columns_with_missing_values].isnull().sum()
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# To hold variable names

labels = [] 



# To hold the count of missing values for each variable 

valuecount = [] 



# To hold the percentage of missing values for each variable

percentcount = [] 



for col in columns_with_missing_values:

    labels.append(col)

    valuecount.append(train[col].isnull().sum())

    # crystallizer.shape[0] will give the total row count

    percentcount.append(train[col].isnull().sum()/train.shape[0])



ind = np.arange(len(labels))



fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,18))



rects = ax1.barh(ind, np.array(valuecount), color='blue')

ax1.set_yticks(ind)

ax1.set_yticklabels(labels, rotation='horizontal')

ax1.set_xlabel("Count of missing values")

ax1.set_title("Variables with missing values")



rects = ax2.barh(ind, np.array(percentcount), color='pink')

ax2.set_yticks(ind)

ax2.set_yticklabels(labels, rotation='horizontal')

ax2.set_xlabel("Percentage of missing values")

ax2.set_title("Variables with missing values")

plt.tight_layout()
# Check which variables have missing values



columns_with_missing_values = test.columns[test.isnull().any()]

test[columns_with_missing_values].isnull().sum()
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# To hold variable names

labels = [] 



# To hold the count of missing values for each variable 

valuecount = [] 



# To hold the percentage of missing values for each variable

percentcount = [] 



for col in columns_with_missing_values:

    labels.append(col)

    valuecount.append(test[col].isnull().sum())

    # crystallizer.shape[0] will give the total row count

    percentcount.append(test[col].isnull().sum()/test.shape[0])



ind = np.arange(len(labels))



fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,18))



rects = ax1.barh(ind, np.array(valuecount), color='blue')

ax1.set_yticks(ind)

ax1.set_yticklabels(labels, rotation='horizontal')

ax1.set_xlabel("Count of missing values")

ax1.set_title("Variables with missing values")



rects = ax2.barh(ind, np.array(percentcount), color='pink')

ax2.set_yticks(ind)

ax2.set_yticklabels(labels, rotation='horizontal')

ax2.set_xlabel("Percentage of missing values")

ax2.set_title("Variables with missing values")

plt.tight_layout()
!pip install pandas-profiling[notebook,html]
from pandas_profiling import ProfileReport

profile = ProfileReport(train)
profile
def tokenization(text):

    word = text.split()

    return word

train['text'] = train['text'].apply(tokenization)

test['text'] = test['text'].apply(tokenization)
train['text']
def lowercase(word):

    new_word = list()

    for i in word:

        i = i.lower()

        new_word.append(i)

    return new_word

train['text'] = train['text'].apply(lowercase)

test['text'] = test['text'].apply(lowercase)    
train['text'] 
def remove_punctuations(word):

    new_word = list() 

    for i in word:

        for j in s.punctuation:

            i = i.replace(j,'')

        new_word.append(i)

    return new_word

train['text'] = train['text'].apply(remove_punctuations)

test['text'] = test['text'].apply(remove_punctuations)            
train['text']
def remove_numbers(word):

    no_num_word = list()

    new_word = list()

    for i in word:

        for j in s.digits:    

            i = i.replace(j,'')

        no_num_word.append(i)

    for i in no_num_word:

        if i!='':

            new_word.append(i)

    return new_word

train['text'] = train['text'].apply(remove_numbers)

test['text'] = test['text'].apply(remove_numbers)    
train['text']
def remove_stopwords(word):

    stop_words_ = stopwords.words('english')

    new_word = list()

    for i in word:

        if i not in stop_words_:

            new_word.append(i)

    return new_word

train['text'] = train['text'].apply(remove_stopwords)

test['text'] = test['text'].apply(remove_stopwords)  
train['text']
def remove_spaces(word):

    new_word = list()

    for i in word:

        i = i.strip()

        new_word.append(i)

    return new_word

train['text'] = train['text'].apply(remove_spaces)

test['text'] = test['text'].apply(remove_spaces)  
train['text']
# Stemmer

stemmer = nltk.stem.PorterStemmer()

def stemming(word):

    new_word = list()

    for i in word:

        i = stemmer.stem(i)

        new_word.append(i)

    return new_word

train['text'] = train['text'].apply(stemming)

test['text'] = test['text'].apply(stemming)  
train['text']
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatization(word):

    new_word = list()

    for i in word:

        i = lemmatizer.lemmatize(i)

        new_word.append(i)

    return new_word

train['text'] = train['text'].apply(lemmatization)

test['text'] = test['text'].apply(lemmatization)  
train['text']
train['text'] = train['text'].apply(lambda x: ''.join(i+' ' for i in x))

test['text'] = test['text'].apply(lambda x: ''.join(i+' ' for i in x))
train.head()
#A disaster tweet

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
#No disaster tweet

no_disaster_tweets = train[train['target']==0]['text']

no_disaster_tweets.values[1]
from wordcloud import WordCloud

fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=[23, 10])

wordcloud1 = WordCloud(background_color='white',width=500,height=300).generate(" ".join(disaster_tweets))

ax_1.imshow(wordcloud1)

ax_1.axis('off')

ax_1.set_title('Disaster Tweets',fontsize=40);



wordcloud2 = WordCloud(background_color='white',width=500,height=300).generate(" ".join(no_disaster_tweets))

ax_2.imshow(wordcloud2)

ax_2.axis('off')

ax_2.set_title('Non Disaster Tweets',fontsize=40);
count_vectorizer = CountVectorizer()

train_count = count_vectorizer.fit_transform(train['text'])

test_count = count_vectorizer.transform(test['text'])
# gnb = GaussianNB()

# def select_model(x,y,model):

#     scores = cross_val_score(model, x, y, cv=5, scoring='f1')

#     acc = np.mean(scores)

#     return acc

# select_model(train_count.toarray(),train['target'], gnb)
logistic_count = LogisticRegression()

scores = model_selection.cross_val_score(logistic_count, train_count.toarray(), train["target"], cv=5, scoring="f1")

scores
# logistic = LogisticRegression()

# select_model(train_count.toarray(),train['target'], logistic)
# multinomial_NB = MultinomialNB()

# select_model(train_count.toarray(),train['target'],multinomial_NB)
multinomial_NB_count = MultinomialNB()

scores = model_selection.cross_val_score(multinomial_NB_count, train_count.toarray(), train["target"], cv=5, scoring="f1")

scores
gaussian_nb_count = GaussianNB()

scores = model_selection.cross_val_score(gaussian_nb_count, train_count.toarray(), train["target"], cv=5, scoring="f1")

scores
rf_count = RandomForestClassifier(n_estimators = 10)

scores = model_selection.cross_val_score(rf_count, train_count.toarray(), train["target"], cv=5, scoring="f1")

scores
# import xgboost as xgb

# clf_xgb_count = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

#                         subsample=0.8, nthread=10, learning_rate=0.1)

# scores = model_selection.cross_val_score(clf_xgb_count, train_count.toarray(), train["target"], cv=5, scoring="f1")

# scores
tfidf = TfidfVectorizer()

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test['text'])
logistic_tfidf = LogisticRegression()

scores = model_selection.cross_val_score(logistic_tfidf, train_tfidf.toarray(), train["target"], cv=5, scoring="f1")

scores
multinomial_NB_tfidf = MultinomialNB()

scores = model_selection.cross_val_score(multinomial_NB_tfidf, train_tfidf.toarray(), train["target"], cv=5, scoring="f1")

scores
gaussian_nb_tfidf = GaussianNB()

scores = model_selection.cross_val_score(gaussian_nb_tfidf, train_tfidf.toarray(), train["target"], cv=5, scoring="f1")

scores
# select_model(train2.toarray(),train['target'],gnb)
# select_model(train2.toarray(),train['target'],lr)
# clf_NB = MultinomialNB()

# select_model(train1.toarray(),train['target'],clf_NB)
# rf = RandomForestClassifier(n_estimators = 10)

# select_model(train_tfidf.toarray(),train['target'],rf)
rf_tfidf = RandomForestClassifier(n_estimators = 10)

scores = model_selection.cross_val_score(rf_tfidf, train_tfidf.toarray(), train["target"], cv=5, scoring="f1")

scores
# import xgboost as xgb

# clf_xgb_tfidf = xgb.XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.1)

# scores = model_selection.cross_val_score(clf_xgb_tfidf, train_tfidf.toarray(), train["target"], cv=5, scoring="f1")

# scores
multinomial_NB_count.fit(train_count.toarray(),train['target'])

predictions = multinomial_NB_count.predict(test_count.toarray())

pred[:15]
df_submission = {"id": test['id'],

                 "target": predictions}

submission = pd.DataFrame(df_submission)

submission.to_csv('submission_df.csv',index=False)