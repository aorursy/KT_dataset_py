# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import seaborn as sns

import string

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, cross_val_score
df = pd.read_csv('/kaggle/input/us-state-of-the-union-addresses-1790-2019/state_ofthe_union_texts.csv')
df.head()
df.info()
df.describe()
len(df['President'].unique())
df['President'].unique()
df['Text Length'] = df['Text'].apply(len)

df['Word Count'] = df['Text'].apply (lambda t: len(nltk.tokenize.wordpunct_tokenize(t)))
df.head()
sns.distplot(df['Text Length'], bins=10)

plt.title("Speeches's Text Length Frequency Distribution")
pd.cut(df['Text Length'], bins=10).value_counts()
sns.distplot(df['Word Count'], bins=10)

plt.title("Speeches's Word Count Frequency Distribution")
pd.cut (df['Word Count'], bins=10).value_counts()
sns.distplot(df['Year'], bins=10)

plt.title("Speeches's Year Frequency Distribution")
pd.cut (df['Year'], bins=10).value_counts()
plt.figure(figsize=(15,10))

ax = sns.boxplot(x='President', y='Text Length', data=df, palette='inferno')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.title('Text Length Plot by President')
g = sns.factorplot('President', data=df, kind='count', aspect=3, palette='inferno')

g.set_xticklabels(rotation=90)

plt.title('Number of Speeches by President')
df = df[(df['President'] != 'Zachary Taylor') & (df['President'] != 'Warren G. Harding') & (df['President'] != 'Ronald Reagan')]
pipeline = Pipeline([

    ('cv', CountVectorizer(stop_words='english', strip_accents='ascii')),

    ('lsvm', SGDClassifier(loss='hinge', verbose=0, penalty='l1'))

])
data_train= []

for p in df['President'].unique():

    text1 = df[df['President'] == p].sample(1)['Text'].values[0]

    data_train.append([p, text1])

    text2 = ''

    while True:

        text2 = df[df['President'] == p].sample(1)['Text'].values[0]

        if text2 != text1:

            break

    data_train.append([p, text2])

df_train= pd.DataFrame(data_train, columns=['President', 'Text'])
%time pipeline.fit(df_train['Text'], df_train['President'])
%time preds = pipeline.predict(df['Text'])
confusion_matrix(df['President'], preds)
print (classification_report(df['President'], preds))