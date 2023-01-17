import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
dir = '/kaggle/input/nlp-getting-started'

print(dir)
train_master = pd.read_csv(os.path.join(dir,'train.csv'))

test_master = pd.read_csv(os.path.join(dir,'test.csv'))
print(train_master.head())
train = train_master.copy()

test = test_master.copy()
train['keyword'].unique()
print(len(train['keyword'].unique()))

print(len(test['keyword'].unique()))
train['keyword'].loc[train['keyword'].notna()] = train['keyword'].loc[train['keyword'].notna()].apply(lambda x: ' '.join(x.split('%20')))

test['keyword'].loc[test['keyword'].notna()] = test['keyword'].loc[test['keyword'].notna()].apply(lambda x: ' '.join(x.split('%20')))
print(train.shape)

split = train.shape[0]

temp = train.append(test, sort=False)
print(temp.head())
from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 

   

ps = PorterStemmer()



temp['keyword'].loc[temp['keyword'].notna()] = temp['keyword'].loc[temp['keyword'].notna()].apply(lambda x: ps.stem(x))
train['keyword'] = temp['keyword'][:split]

test['keyword'] = temp['keyword'][split:]
print(train.shape)

print(test.shape)
print(len(train['keyword'].unique()))

print(len(test['keyword'].unique()))
print(set(train['keyword']).symmetric_difference(set(test['keyword'])))
print(set(train['keyword']) - set(test['keyword']))
print(set(test['keyword']) - set(train['keyword']))
print(train['keyword'].unique())
fig, ax = plt.subplots(figsize=(40,8))

sns.countplot(x="keyword", hue="target", data=train, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

fig.show()
train['keyword'] = np.where(train['keyword'].isna(),'None',train['keyword'])

test['keyword'] = np.where(test['keyword'].isna(),'None',test['keyword'])
# from sklearn.feature_extraction.text import TfidfVectorizer



# vectorizer = TfidfVectorizer()



# train['keyword'] = vectorizer.fit_transform(train['keyword']).toarray()

# test['keyword'] = vectorizer.fit_transform(test['keyword']).toarray()
X_train = train['keyword']

y_train = train['target']



X_test = test['keyword']
temp = X_train.append(X_test)

temp = pd.get_dummies(temp)



print(temp.shape)
X_train = temp.iloc[:7613]

X_test = temp.iloc[7613:]
print(X_train.shape)

print(y_train.shape)
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()



gnb.fit(X_train, y_train)
gnb.score(X_train, y_train)
predictions = gnb.predict(X_test)
submission = pd.DataFrame()

submission['id'] = test['id']
submission['target'] = predictions
print(submission.head())
submission.to_csv('../working/submission.csv', encoding='utf-8', index=False)