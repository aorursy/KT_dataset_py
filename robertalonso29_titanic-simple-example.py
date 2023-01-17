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



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
%matplotlib inline

%config InlineBackend.figure_format='retina'



import seaborn as sns

sns.set()
sns.distplot(train.Age.dropna());
X, y = train.drop('Survived', axis=1), train.Survived

X_test = test.copy()
X.describe(include='all').T
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']

keep_cols = [c for c in X.columns if c not in drop_cols]



X, X_test = X[keep_cols].copy(), X_test[keep_cols].copy()
X[keep_cols].head()
cat_cols = ['Sex', 'Embarked']

cont_cols = [c for c in keep_cols if c not in cat_cols]
cat_cols, cont_cols
from sklearn.preprocessing import LabelEncoder



def preprocess_cats(X, cat_cols, lencoders=None):

    train = lencoders is None

    # Initialize label encoders

    if train:

        lencoders = {}

    for c in cat_cols:

        # Parse to string before encoding

        X[c] = X[c].astype(str)

        if train: 

            lencoders[c] = LabelEncoder().fit(X[c])

        X[c] = lencoders[c].transform(X[c])

    return lencoders



def preprocess_conts(X, cont_cols, means=None):

    train = means is None

    # Initialize mean dicts

    if train:

        means = {}

    for c in cont_cols:

        if train:

            means[c] = {}

            means[c]['mean'] = X[c].mean()

        # Create missing column only if there are 

        # more than 15 null rows in the training set

        if train and X[c].isnull().sum() > 15:

            means[c]['isnull'] = True

        if 'isnull' in means[c]:

            X[f'{c}Missing'] = X[c].isnull()

        X[c] = X[c].fillna(means[c]['mean'])

    return means
lencoders = preprocess_cats(X, cat_cols)

preprocess_cats(X_test, cat_cols, lencoders);



means = preprocess_conts(X, cont_cols)

preprocess_conts(X_test, cont_cols, means);
X.head()
X_test.head()
X.describe().T
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X, y)
y_pred = tree.predict(X)

y_test = tree.predict(X_test)
pd.Series(y_test).value_counts()
submission.Survived.value_counts()
submission.tail()
submission['Survived'] = y_test

submission.to_csv('submission.csv', index=False);
submission.head()
from sklearn.metrics import accuracy_score

accuracy_score(y, y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y, y_pred)
sns.heatmap(confusion_matrix(y, y_pred), cmap='Blues', annot=True, fmt='d');