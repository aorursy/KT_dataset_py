#importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
print(true.info())

print('='*50)

print(fake.info())
true['target'] = 1

fake['target'] = 0

df = pd.concat([true,fake], ignore_index=True)

df.info()
df['combined'] = df['subject'] + df['title'] + df['text']
df['combined'] = df.text.apply(lambda x: x.lower())

display(df.head())
X_train, X_test, y_train, y_test = train_test_split(df.combined, df.target, test_size=.25, random_state=123, stratify=df.target)
for i in [X_train, X_test, y_train, y_test]:

    print(i.shape)

    print('\n')
print('y_train distribution:')

print(y_train.value_counts())

print('y_test distribution:')

print(y_test.value_counts())
pipeline = Pipeline([('vect',CountVectorizer(stop_words='english')),

                     ('model',LogisticRegression())])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print('accuracy: {:.2f}%'.format(accuracy_score(y_test,pred)*100))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm, cmap = 'Blues', annot= True, fmt = 'd', xticklabels = ['fake','real'], yticklabels = ['fake','real'])

plt.show()
print(classification_report(y_test,pred, target_names=['fake','real']))
print(accuracy_score(y_test,pred))
pipeline = Pipeline([('vect',CountVectorizer(stop_words='english')),

                     ('model',MultinomialNB())])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print('accuracy: {:.2f}%'.format(accuracy_score(y_test,pred)*100))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm, cmap = 'Blues', annot= True, fmt = 'd', xticklabels = ['fake','real'], yticklabels = ['fake','real'])

plt.show()
print(classification_report(y_test,pred, target_names=['fake','real']))
print(accuracy_score(y_test,pred))