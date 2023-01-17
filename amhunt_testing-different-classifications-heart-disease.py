import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.naive_bayes import MultinomialNB
# Load in data

df = pd.read_csv('../input/heart.csv')
df.head(10)
# Convert target to bool

df['target'] = df['target'].astype('bool')
df.dtypes
# Summary Statistics

df.describe().round(decimals = 2)
# Correlation

df.corr()
# Correlation heatmap

f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, linewidths=0.5, fmt='.1f', ax=ax)

plt.show()
# Look for missing data

df.info()
# cp has the strongest correlation with target, so let's look at that

df['cp'].value_counts()
# And thalach

df['thalach'].plot(kind="hist", bins=20)
# Split training and testing

df_x = df.drop(['target'], axis=1)

x = df_x.values

y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
# Train the model

gnb = MultinomialNB()

gnb.fit(x_train, y_train)
# Test accuracy

print('Naive Bayes Score: %.3f' % gnb.score(x_test,y_test))
# Train the model

dt = tree.DecisionTreeClassifier()

dt.fit(x_train, y_train)
# Test accuracy

print('Decision Tree Score: %.3f' % dt.score(x_test,y_test))
# Train the model

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
# Test accuracy

print('Random Forest Score: %.3f' % rfc.score(x_test,y_test))