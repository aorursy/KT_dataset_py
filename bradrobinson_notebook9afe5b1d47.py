# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.columns)

train.head(10)
from sklearn.ensemble import RandomForestClassifier

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']



clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2)

clf.fit(train[columns], train['Survived'])

prediction = clf.predict(test[columns])
train.head()
def categorize_data(df, col):

    df[col] = pd.Series(df[col]).astype("category").cat.codes



def categorize_cols(cols, train, test):

    for col in cols:

        categorize_data(train, col)

        categorize_data(test,col)

str_cols = ['Sex', 'Embarked']

categorize_cols(str_cols, train, test)
test.head()
from sklearn.metrics import roc_auc_score 

clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2)

clf.fit(train[columns], train['Survived'])

prediction = clf.predict(test[columns])

print(test['Survived'], prediction)
print(train[train['Age'].isnull()])
columns.append('Survived')

train = train[columns].dropna()

columns.pop()

test = test[columns].dropna()

print(train.shape[0], test.shape[0])
from sklearn.metrics import roc_auc_score 

clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2)

clf.fit(train[columns], train['Survived'])

prediction = clf.predict(test[columns])

print(len(prediction))
prediction = clf.predict(train[columns])

print(roc_auc_score(train['Survived'], prediction))
train_u = pd.read_csv("../input/train.csv")

train_age = train['Age'].dropna()

train_age_mean = train_age.mean()

test_u = pd.read_csv("../input/test.csv")

test_age = test['Age'].dropna()

test_age_mean = test_age.mean()

print(train_age_mean, test_age_mean)
def change_age(row):

    if row > 0:

        return row

    else:

        return train_age_mean



train_u['Age'] = train_u["Age"].apply(change_age)

train_age_mean = test_age_mean

test_u['Age'] = test_u['Age'].apply(change_age)
categorize_cols(columns, train_u, test_u)
clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2)

clf.fit(train_u[columns], train_u['Survived'])

prediction = clf.predict(test_u[columns])

print(len(prediction))
train_prediction = clf.predict(train_u[columns])

print(roc_auc_score(train_u['Survived'], train_prediction))
pred_df = pd.DataFrame(prediction, columns = ['Prediction'])

pred_df.to_csv("predictions.csv")