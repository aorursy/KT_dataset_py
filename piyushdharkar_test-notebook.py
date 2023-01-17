# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS 



from scipy.sparse import hstack



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 

from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PATH = '../input/'
train = pd.read_csv(PATH + 'train-balanced-sarcasm.csv')

test = pd.read_csv(PATH + 'test-unbalanced.csv')
train.head(10)
train.isna().sum()
train_without_na = train.dropna(subset=['comment'])
train_without_na.info()
train.describe()
train_without_na.shape
train_without_na['label'].value_counts()
train_without_na['author'].value_counts()
train_without_na.groupby('author')['label'].agg([np.size, np.mean, np.sum]).sort_values(by='sum', ascending=False)
train_without_na.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum]).sort_values(by='sum', ascending=False)
fig, ax = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(15, 12))



feature_id = 0



feature_list = ['score', 'ups', 'downs']



for row in ax:

    feature = feature_list[feature_id]

    

    axes = row.flatten()

    train_without_na.hist(ax=axes, column=feature, by='label', bins=50, log=True)



    axes[0].set_xlabel(feature)

    axes[0].set_ylabel('Log Frequency')



    axes[1].set_xlabel(feature)

    axes[1].set_ylabel('Log Frequency')



    feature_id += 1
sarcastic_comments = str(train_without_na[train_without_na['label'] == 1]['comment'])

plt.figure(figsize=(12, 12))

word_cloud = WordCloud(stopwords=STOPWORDS)

word_cloud.generate(sarcastic_comments)

plt.imshow(word_cloud)
sincere_comments = str(train_without_na[train_without_na['label'] == 0]['comment'])

plt.figure(figsize=(12, 12))

word_cloud = WordCloud(stopwords=STOPWORDS)

word_cloud.generate(sincere_comments)

plt.imshow(word_cloud)
train_removed_features = train_without_na.iloc[:, :-3].drop('author', axis=1)

train_removed_features.head(10)
train_x, train_y = train_removed_features.drop('label', axis=1), train_removed_features[['label']]
Counter([word for word in str(train_x['comment']).split(' ') if word not in STOPWORDS])
tfidf_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=None)

comment_sparse = tfidf_comment.fit_transform(train_x['comment'])
tfidf_subreddit = TfidfVectorizer(ngram_range=(1, 1), max_features=None)

subreddit_sparse = tfidf_comment.fit_transform(train_x['subreddit'])
print(len(tfidf_comment.vocabulary_))
cont_var = train_x.iloc[:,-3:]

scaler = StandardScaler()

scaled_cont_var = scaler.fit_transform(cont_var)
train_x_sparse = hstack([comment_sparse, subreddit_sparse, scaled_cont_var])
x_train, x_test, y_train, y_test = train_test_split(train_x_sparse, train_y)
clf = LogisticRegression(solver='liblinear', verbose=True)
clf.fit(x_train, y_train)
clf.score(x_train, y_train)
clf.score(x_test, y_test)
y_pred = clf.predict(x_test)
cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
sns.heatmap(cf_matrix, cmap='Blues')