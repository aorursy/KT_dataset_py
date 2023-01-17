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

import re

import nltk

from nltk.corpus import stopwords

import seaborn as sns

import matplotlib.pyplot as plt

import string

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.naive_bayes import MultinomialNB



# Any results you write to the current directory are saved as output.
### Reading train dataset ###



train = pd.read_csv('../input/nlp-getting-started/train.csv')

train.head()
train.shape
# Reading test data ###

test = pd.read_csv('../input/nlp-getting-started/test.csv')

test.head()
test.shape
train.isnull().sum()
test.isnull().sum()
train.duplicated().sum()
train.target.value_counts()
ax = sns.countplot(x="target", data = train)

plt.show()
print("Example of a disaster tweet:\n")

dis_tw = train.loc[train.target == 1,['text']]

dis_tw.loc[0]
print("Example of a normal tweet:\n")

norm_tw = train.loc[train.target == 0,['text']]

norm_tw.values[0]
loca = train.location.value_counts(dropna = False)

fig = plt.figure(figsize = (16, 8))

sns.barplot(x= loca.iloc[0:31], y = loca.index[0:31], orient ='h')

plt.title("Top 30 locations in the train set")
# country_code = {"New York" :"USA", "United States": "USA", "London":"UK", "Los Angeles, CA": "USA", "Mumbai": "India",

#                "Washington, DC":"USA", "Chicago":"USA", "Chicago, IL":'USA',"California":'USA',"California, USA":'USA',

#                             "FLorida":'USA',

#                             "Nigeria":'Africa',

#                             "Kenya":'Africa',

#                             "Everywhere":'Worldwide',"San Francisco":'USA',

#                             "United Kingdom":'UK',"Los Angeles":'USA',

#                             "Toronto":'Canada',"San Francisco, CA":'USA',

#                             "NYC":'USA',

#                              "Earth":'Worldwide',"Ireland":'UK',

#                              "New York, NY":'USA'}

# train.location = train.location.map(country_code)
loca = train.location.unique()

# sns.barplot(x= loca.values[0:4], y = loca.index[0:4], orient ='h')

# plt.title("Top 30 locations in the train set")

len(loca)
c = train.keyword.value_counts(dropna = False)

c
sns.barplot(y = c.index[0:11], x = c.iloc[0:11], orient = 'h')

plt.title("Top 10 keywords in train set")
### Identifying top 10 keywords for disaster and non-disaster tweets ###

disaster = train.loc[train.target == 1]

top_d = disaster['keyword'].value_counts()

normal = train.loc[train.target == 0]

top_n = normal['keyword'].value_counts()

fig = plt.figure(figsize = (15,6))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.barplot(y = top_d.index[0:11], x = top_d.iloc[0:11], ax = ax1, orient ='h' , color = 'red')

ax1.set_title("Top 10 keywords in disaster tweets")

sns.barplot(y = top_n.index[0:11], x = top_n.iloc[0:11], ax = ax2, orient ='h', color = 'pink' )

ax2.set_title("Top 10 keywords in normal tweets")
disaster.shape
tweet_char_d = disaster.text.str.len()

tweet_char_d.sort_values(ascending = False)

tweet_char_n = normal.text.str.len()

fig = plt.figure(figsize = (12,6))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1.hist(tweet_char_d, color = 'red')

ax1.set_title("Char in disaster tweets")

ax2.hist(tweet_char_n, color = 'pink')

ax2.set_title("Char in normal tweets")
string.punctuation
type(string.punctuation)
def preprocess(element):

    

    ### convert all to lowercase ###

    element = element.lower()

    

    ### get rid of punctuation ###

    element = element.translate({ord(i): None for i in string.punctuation})

        

    

    ### get rid of weblinks ###

    element = re.sub('https?://\S+|www\.\S+', '', element)

    

    return element





train['text'] = train.text.map(preprocess)

test['text'] = test.text.map(preprocess)

train['text'].head(10)
fig = plt.figure(figsize = (12,6))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

wc1 = WordCloud(background_color='white',width=  600,

                        height=500).generate(" ".join(disaster.text))

ax1.imshow(wc1)

ax1.set_title("Disaster tweets")

ax1.axis('off')

wc2 = WordCloud(background_color='white',width=600,

                        height=500).generate(" ".join(normal.text))

ax2.imshow(wc2)

ax2.set_title("Normal tweets")

ax2.axis('off')


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].map(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].map(lambda x: tokenizer.tokenize(x))

train['text'].head()
print(stopwords.words('english'))
def stopwords_remove(text):

         words = [i for i in text if i not in stopwords.words('english')]

         return words

train.text = train.text.map(stopwords_remove)

test.text = test.text.map(stopwords_remove)

train.head()
### combining list of strings ###



def combine(text_list):

    joined_text = ' '.join(text_list)

    return joined_text



train['text'] = train['text'].map(combine)

test['text'] = test['text'].map(combine)

train.head()
### Using CountVectorizer ###



countvec = CountVectorizer()

train_vec = countvec.fit_transform(train['text'])

test_vec = countvec.transform(test["text"])
### TFIDF ###



tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
test_tfidf
lr = LogisticRegression(random_state = 42)

model = Pipeline([('classification', lr)])

param = {'classification__C': [0.001, 0.01, 0.1,1,  10, 100]}

grid_search_log = GridSearchCV(model, param, scoring ='f1',refit=True,  cv= 5)

grid_search_log.fit(train_vec,train['target'])

print(grid_search_log.best_estimator_)

log_best = grid_search_log.best_estimator_

resultsdf = pd.DataFrame(grid_search_log.cv_results_)

resultsdf
print("mean test score for Logistic Regression with Counts:", resultsdf.loc[resultsdf['rank_test_score']==1, ["mean_test_score"]])
lr = LogisticRegression(random_state = 42)

model = Pipeline([('classification', lr)])

param = {'classification__C': [0.001, 0.01, 0.1,1,  10, 100]}

grid_search_log = GridSearchCV(model, param, scoring ='f1',refit=True,  cv= 5)

grid_search_log.fit(train_tfidf,train['target'])

print(grid_search_log.best_estimator_)

log_best = grid_search_log.best_estimator_

resultsdf = pd.DataFrame(grid_search_log.cv_results_)

resultsdf

print("mean test score for Logistic Regression with TFIDF:", resultsdf.loc[resultsdf['rank_test_score']==1, ["mean_test_score"]])
nb = MultinomialNB()

count_scores = cross_val_score(nb, train_vec, train["target"], cv=5, scoring="f1")

count_scores
nb_tfidf = MultinomialNB()

tfidf_scores = cross_val_score(nb_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

tfidf_scores
nb_tfidf.fit(train_tfidf, train["target"])




def save_submission_file(model):

    test_pred = model.predict(test_tfidf)

    dat ={'id':test['id'],  "target":test_pred}

    submission = pd.DataFrame(dat)

    csvfile = submission.to_csv("submission.csv", index=False)

    return csvfile

save_submission_file(nb_tfidf)