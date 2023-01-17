import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')

df.head()

test = pd.read_csv('../input/test.csv')
df['NameLength'] = df['Name'].apply(len)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = df.drop(['Ticket', 'Cabin', 'Embarked', 'Name'], axis=1)

df = df.dropna()

y = df['Survived']

df = df.drop(['Survived'], axis=1)

df.head()
test['NameLength'] = test['Name'].apply(len)

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

test = test.drop(['Ticket', 'Cabin', 'Embarked', 'Name'], axis=1)

test.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df.astype(float).corr(), cmap=colormap, annot=True)
X_train = df.as_matrix()

y_train = y.as_matrix()

X_test = test.dropna().as_matrix()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



logreg.score(X_train, y_train)
svm = SVC()

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

svm.score(X_train, y_train)