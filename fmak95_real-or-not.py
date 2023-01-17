#Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

import matplotlib.pyplot as plt

import string



from collections import Counter

from nltk.corpus import stopwords

import re



# from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import f1_score
#Take a peak at the data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
train.info()
#Lets answer our first question... whats the distribution of real to fake look like?

tmp = train.groupby('target').count()

sns.barplot(tmp.index, tmp.id)



print("Number of real disasters: {}".format(tmp[tmp.index == 1].id.values[0]))

print("Number of fake disasters: {}".format(tmp[tmp.index == 0].id.values[0]))
def find_num_hashtags(s):

    arr = s.split()

    ans = len([word for word in arr if word[0] == '#'])

    return ans
#Create some new columns I think would be useful: num_words, num_hashtags, num_characters

train['num_words'] = train.text.apply(lambda s: len(s.split()))

train['num_hashtags'] = train.text.apply(find_num_hashtags)

train['num_characters'] = train.text.apply(lambda s: len(s))
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train[train.target == 1].num_words)

plt.title('Distribution of num_words in case of real disaster.')



plt.subplot(1,2,2)

sns.distplot(train[train.target == 0].num_words)

plt.title('Distribution of num_words in case of fake disaster.')
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train[train.target == 1].num_hashtags)

plt.title('Distribution of num_hashtags in case of real disaster.')



plt.subplot(1,2,2)

sns.distplot(train[train.target == 0].num_hashtags)

plt.title('Distribution of num_hashtags in case of fake disaster.')
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train[train.target == 1].num_characters)

plt.title('Distribution of num_characters in case of real disaster.')



plt.subplot(1,2,2)

sns.distplot(train[train.target == 0].num_characters)

plt.title('Distribution of num_characters in case of fake disaster.')
# Count keywords

keywords = []

i = 0

for word in train.keyword:

    if word is not np.nan:

        keywords.append(word)



keyword_freq = Counter(keywords)
print("For the top 10 most occuring keywords:")

print('________________________')

for i in range(10):

    word = keyword_freq.most_common(10)[i][0]

    print("Keyword: {}".format(word))

    print(">>>> Num of occurences in REAL disasters = {}".format(len(train[(train.keyword == word) & (train.target == 1)])))

    print(">>>> Num of occurences in FAKE disasters = {}".format(len(train[(train.keyword == word) & (train.target == 0)])))

    print("__________________________________")
# how many tweets containing a specific keyword are real vs fake?

for word in keyword_freq:

    ratio = len(train[(train.keyword == word) & (train.target == 1)]) / len(train[train.keyword == word])

    train.loc[train.keyword == word, 'keyword_ratio'] = ratio

    

keyword_ratio = train[['keyword','keyword_ratio']].drop_duplicates()
# Top ten keywords with the highest ratio

keyword_ratio.sort_values('keyword_ratio', ascending=False).head(10)
def remove_url(s):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',s)



def remove_punctuation(s):

    table = str.maketrans("","",string.punctuation)

    return s.translate(table)



def remove_emoji(s):

    return s.encode('ascii', 'ignore').decode('ascii')



train['text_cleaned'] = train.text.apply(remove_url)

train['text_cleaned'] = train.text_cleaned.apply(remove_punctuation)

train['text_cleaned'] = train.text_cleaned.apply(remove_emoji)
tf_idf_vectorizer = TfidfVectorizer()

X = tf_idf_vectorizer.fit_transform(train['text_cleaned'])

X_train, X_val, y_train, y_val = train_test_split(X, train['target'], test_size=0.25)
clf = RidgeClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_val)

score = f1_score(predictions, y_val)

print(score)
scores = cross_val_score(clf, X, train["target"], cv=5, scoring="f1")

scores