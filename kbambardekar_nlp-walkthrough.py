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
import numpy as np 
import pandas as pd 
import re
import string
import nltk
from nltk.corpus import stopwords
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import model_selection, feature_extraction, linear_model
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Reading in data
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# Dimensions of train data
print("Train Dimensions: ", train.shape)
# Dimensions of test data
print("Test Dimensions: ", test.shape)
# Viewing first 5 rows of train data
train.head()
# Viewing first 5 rows of test data
test.head()
train.isnull().sum()
test.isnull().sum()
# Creating dataframe with percentage of missing values for train data

train_perc_missing = train.isnull().mean()*100
percentage_missing = pd.DataFrame({'Train Missing Percentage': train_perc_missing.sort_values(ascending=False)})
percentage_missing
# Creating dataframe with percentage of missing values for test data

test_perc_missing = test.isnull().mean()*100
percentage_missing = pd.DataFrame({'Test Missing Percentage': test_perc_missing.sort_values(ascending=False)})
percentage_missing
# Selecting 'text' values that are non-disastrous
non_disastrous = train[train['target']==0]['text']

# I inputted 4 to select the 4th row of the non-disastrous values
non_disastrous.values[4]
# Selecting 'text' values that are disastrous
disastrous = train[train['target']==1]['text']

# I inputted 2 to select the 2th row of the disastrous values
disastrous.values[2]
# Viewing number of disastrous and non disastrous tweets
train['target'].value_counts()
# Plotting number of disastrous and non disastrous tweets
sns.barplot(train['target'].value_counts().index, train['target'].value_counts())
# Checking number of unique values in 'keyword' feature
train['keyword'].nunique()
# Plotting the first 20 most common keywords
figure = plt.figure(figsize=(14,12))
sns.barplot(y=train['keyword'].value_counts().index[:20], x=train['keyword'].value_counts()[:20])
print(train['location'].nunique())
# Viewing first 20 most common locations
figure = plt.figure(figsize=(14,12))
sns.barplot(y=train['location'].value_counts().index[:20], x=train['location'].value_counts()[:20])
# Viewing last 20 least common locations
figure = plt.figure(figsize=(14,12))
sns.barplot(y=train['location'].value_counts().index[-20:], x=train['location'].value_counts()[-20:])
def lowercase_text(text):
    text = text.lower()
    return text

train['text'] = train['text'].apply(lambda x: lowercase_text(x))
test['text'] = test['text'].apply(lambda x: lowercase_text(x))
train['text'].head()
train['text'].head()
# Removing punctuation, html tags, symbols, numbers, etc.
def remove_noise(text):
    # Dealing with Punctuation
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
# Calling remove_noise function in order to remove noise
train['text'] = train['text'].apply(lambda x: remove_noise(x))
test['text'] = test['text'].apply(lambda x: remove_noise(x))
train['text'].head(20)
#Remove StopWords

!pip install nlppreprocess
from nlppreprocess import NLP

nlp = NLP()

train['text'] = train['text'].apply(nlp.process)
test['text'] = test['text'].apply(nlp.process)  
train['text'].head(20)
#Stemming
stemmer = SnowballStemmer("english")

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

train['text'] = train['text'].apply(stemming)
test['text'] = test['text'].apply(stemming)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
fig, (ax1) = plt.subplots(1, figsize=[24,20])
wordcloud = WordCloud( background_color='white',
                        width=600,
                        height=600).generate(" ".join(train['text']))
ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=16);
# Using CountVectorizer to change the teweets to vectors
count_vectorizer = CountVectorizer(analyzer='word', binary=True)
count_vectorizer.fit(train['text'])

train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])


# Printing first vector
print(train_vectors[0].todense())
y = train['target']
# Creating a simple MultinomialNB model
model = MultinomialNB(alpha=1)

# Using cross validation to print out our scores
scores = model_selection.cross_val_score(model, train_vectors, y, cv=3, scoring="f1")
scores
# Training model with train_vectors and target variable
model.fit(train_vectors, y)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# Predicting model with the test data that was vectorized (test_vectors)
sample_submission['target'] = model.predict(test_vectors)


# Viewing submission
sample_submission.head()
# Submission
sample_submission.to_csv("submission.csv", index=False)
