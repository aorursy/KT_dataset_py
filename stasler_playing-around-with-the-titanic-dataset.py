# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import re
# Any results you write to the current directory are saved as output.
# read data
train_data = pd.read_csv('../input/train.csv', index_col = 'PassengerId')
test_data = pd.read_csv('../input/test.csv', index_col = 'PassengerId')

# show some basic stats
print('training data: ', train_data.shape)
print('testing data: ', test_data.shape)

print('columns: ', train_data.columns)

train_data.head(10)
def new_features(df):
    # _groupSize = SibSp + Parch + 1
    # no persons traveling on the same ticket
    df['_groupSize'] = df.SibSp + df.Parch + 1
    # _pricePrPerson = Fare / _groupSize
    # Fare is price for whole ticket, so adjusting to ticket price pr person
    df['_pricePrPerson'] = df.Fare / df._groupSize
    return df

def drop_columns(df):
    return df.drop(columns=['Name','SibSp','Parch','Ticket','Fare','Cabin'])

def complete_df(df):
    df.Embarked = df.Embarked.fillna('S')
    return df.fillna(df.median())

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Sex', 'Embarked']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

def apply_classifier(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    acc_score = clf.score(X_train, y_train)
    return acc_score, y_test
# fixing
train_data = complete_df(train_data)
test_data = complete_df(test_data)

# feature engineering
train_data = new_features(train_data)
test_data = new_features(test_data)

train_data[train_data.Age < 5]
test_data[test_data.Fare.isnull()]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.countplot(data=train_data, x='Pclass', hue='Survived',  ax=ax1)
#X_train_df.Pclass.value_counts().sort_index().plot.bar(ax=axarr[0][0])

sns.countplot(data=train_data, x='Embarked', hue='Survived', ax=ax2)
#X_train_df.Embarked.value_counts().plot.bar(ax=axarr[0][1])

sns.distplot(train_data.Age.dropna(), kde=False, ax=ax3)
#X_train_df.Age.plot.hist(ax=axarr[0][2])
# This was no good approach to get the ticket price per person.
# Obviously, persons travelling together are split up to training and test data.
# So counting the number of persons using the same ticket does not work.
# But we do have SibSp and Parch that seem to give the correct group size.
#ticket_price_pr_person = train_data.groupby('Ticket')['Fare'].agg(lambda x: np.mean(x) / np.size(x)).rename('_pricePrPerson')
#train_data.join(ticket_price_pr_person, on='Ticket')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.countplot(train_data._groupSize, ax=ax1)
#X_train_df._groupSize.value_counts().sort_index().plot.bar(ax=axarr[0][2])

sns.distplot(train_data[train_data._pricePrPerson < 200]._pricePrPerson, kde=False, ax=ax2)
#X_train_df[X_train_df._pricePrPerson < 200]._pricePrPerson.plot.hist(ax=axarr[1][0])

(train_data[(train_data._pricePrPerson < 200) & (train_data._groupSize < 5)].sample(100)
# .plot.scatter(x='_groupSize', y='_pricePrPerson', ax=axarr[1][2])
 .plot.hexbin(x='_groupSize', y='_pricePrPerson', gridsize=50, ax=ax3)
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.violinplot(data=train_data, x='Survived', y='Age', ax=ax1)

sns.violinplot(data=train_data[train_data._pricePrPerson < 150], x='Survived', y='_pricePrPerson', ax=ax2)

sns.kdeplot(data=train_data[['Survived','Pclass']], ax=ax3)

sns.pairplot(train_data[['Survived','Pclass','Age','_pricePrPerson','Sex']].dropna(), hue='Sex')
sns.heatmap(train_data.loc[:,['Survived','Pclass','Age','Sex','_groupSize','_pricePrPerson']].corr())

# This does not seem correct... especially Pclass vs Survived

# sns.FacetGrid is cool, see https://www.kaggle.com/startupsci/titanic-data-science-solutions
train_data = drop_columns(train_data)
test_data = drop_columns(test_data)

test_data.describe(include='all')

X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']
X_test = test_data.copy()

X_train, X_test = encode_features(X_train, X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

logreg_clf = LogisticRegression()
acc_logreg, y_logreg = apply_classifier(logreg_clf, X_train, y_train, X_test)
acc_logreg
randomforest_clf = RandomForestClassifier()
acc_randomforest, y_randomforest = apply_classifier(randomforest_clf, X_train, y_train, X_test)
acc_randomforest
svm_clf = SVC()
acc_svm, y_svm = apply_classifier(svm_clf, X_train, y_train, X_test)
acc_svm
submission = pd.DataFrame({'PassengerId': X_test.index, 'Survived': y_randomforest})
submission.to_csv('submission.csv', index=False)