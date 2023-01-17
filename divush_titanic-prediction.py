# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# Any results you write to the current directory are saved as output.
# Beginner's attempt at classification.

df = pd.read_csv('../input/train.csv')

df.head()

for x in ['Name', 'Ticket']:

    del df[x]
# sns.lmplot(x='Age', y='Survived', data=df, fit_reg=False, hue='Pclass')
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

le.get_params()
# Don't need cabin

del df['Cabin']

df.head()
# Couting significance of each feature with survival

cl = list(df.columns)

cl = cl[2:]

cl

for col in cl:

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.countplot(x=col, hue='Survived', data=df)

    plt.xticks(rotation="vertical")

    plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

sns.distplot((df[df['Survived']==0]['Age']).dropna(), ax=ax)

sns.distplot((df[df['Survived']==1]['Age']).dropna(), ax=ax)

plt.show()
# Filling out NAs

df = df.dropna(axis=0, how='any')
# df['Age'].fillna(0.0)

# from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

df['Embarked'] = le1.fit_transform(df['Embarked'])
# Normalize all values

for x in df.columns[2:]:

    df[x] = StandardScaler().fit_transform(df[x].values.reshape(-1, 1))

# df.head()
# Preparing test and train data

X = np.array(df.drop(['PassengerId', 'Survived'], axis=1))

y = np.array(df['Survived'])

df.head()
# A function that tries various params and returns results.

def get_best_model(params):

    result = {}

    for x in params:

        clf = svm.SVC(C=x)

        scores = cross_val_score(clf, X, y, cv=5)

        result[x] = np.mean(scores)

    return result
# RBF kernel SVM

Cvals = [0.01, 0.1, 1, 10, 100]

results = get_best_model(Cvals)

results
# Decision Tree Classifier

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

scores = cross_val_score(clf, X, y, cv=5)

np.mean(scores)
# Printing decision tree

clf.fit(X, y)

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=df.columns[2:], class_names=True)

graph = graphviz.Source(dot_data)

graph
# Logistic Regression

def get_best_model2(params):

    result = {}

    for x in params:

        clf = LogisticRegression(C=x)

        scores = cross_val_score(clf, X, y, cv=5)

        result[x] = np.mean(scores)

    return result
results = get_best_model2(Cvals)

results
# Now, time for submitting code:

testdf = pd.read_csv('../input/test.csv')

testdf

for x in ['Name', 'Ticket', 'Cabin']:

    del testdf[x]

testdf.head()
testdf['Sex'] = le.fit_transform(testdf['Sex'])

testdf['Embarked'] = le1.fit_transform(testdf['Embarked'])
k = np.mean(testdf['Age'].dropna())

testdf['Age'] = testdf['Age'].fillna(k)

k = np.mean(testdf['Fare'].dropna())

testdf['Fare'] = testdf['Fare'].fillna(k)
# for x in testdf.columns[1:]:

#     if sum(testdf[x].isnull())>0:

#         print(str(x) + "  " + str(sum(testdf[x].isnull())))

testdf.columns
for x in testdf.columns[1:]:

    testdf[x] = StandardScaler().fit_transform(testdf[x].values.reshape(-1, 1))
X_train = testdf.drop(['PassengerId'], axis=1)

clf = svm.SVC(C=1)

clf.fit(X, y)

testdf['Survived'] = clf.predict(X_train)

testdf.head()
# testdf.to_csv(path_or_buf='out.csv', columns=[['PassengerID', 'Survived']])

tempdf = testdf[['PassengerId', 'Survived']]

tempdf.to_csv(path_or_buf='out.csv', index=False)