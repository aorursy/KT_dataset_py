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
import os

import pandas as pd

import numpy as np

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.describe()
train[['Sex']].isna().count()
N = train.shape[0]

dummy_M = np.repeat(0, N)

dummy_W = np.repeat(0, N)
man = np.array(train.Sex == 'male')

woman = np.array(train.Sex == 'female')
dummy_M[man] = 1

dummy_W[woman] = 1



SEX = pd.DataFrame({'male' : dummy_M, 'female' : dummy_W})

SEX
train_ = train.drop(['PassengerId', 'Sex' ], axis = 1, inplace = False)

train_.head()
mir_data = pd.concat((train_,SEX),1)

mir_data.head()
#Embarked에 결측치가 있는 행을 뽑기

mir_data.loc[mir_data['Embarked'].isnull()]
mir_data.loc[(mir_data['Pclass'] ==1) | (mir_data['Ticket'] == '113572')]
mir_data.loc[mir_data['Embarked'].isnull()]
from collections import Counter

cnt = Counter(mir_data['Embarked'])
cnt
mir_data.loc[mir_data['Embarked'].isnull()]
mir_data['Embarked'].fillna(value = 'S',inplace = True)
mir_data['Embarked'].isnull().sum()
mir_data.head()
mir_data['Embarked'].unique()
N = train.shape[0]

dummy_S = np.repeat(0, N)

dummy_C = np.repeat(0, N)

dummy_Q = np.repeat(0, N)
S = np.array(train.Embarked == 'S')

C = np.array(train.Embarked == 'C')

Q = np.array(train.Embarked == 'Q')

dummy_S[S] = 1

dummy_C[C] = 1

dummy_Q[Q] = 1



EMBARKED = pd.DataFrame({'S' : dummy_S, 'C' : dummy_C,'Q' : dummy_Q})

EMBARKED
mir_data.drop('Embarked',axis = 1, inplace = True)

mir_data = pd.concat((mir_data,EMBARKED),1)

mir_data.head()
mir_data['Cabin'].value_counts()
cabin_only = mir_data[["Cabin"]].copy()

cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x)
cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)

cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")

cabin_only[cabin_only["Cabin_Data"]]
cabin_only.isnull().sum()
cabin_only[cabin_only["Deck"]=="F"]
cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
cabin_only["Deck"] = cabin_only["Deck"].fillna("N")

cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean())
def one_hot_column(df, label, drop_col=False):

    '''

    This function will one hot encode the chosen column.

    Args:

        df: Pandas dataframe

        label: Label of the column to encode

        drop_col: boolean to decide if the chosen column should be dropped

    Returns:

        pandas dataframe with the given encoding

    '''

    one_hot = pd.get_dummies(df[label], prefix=label)

    if drop_col:

        df = df.drop(label, axis=1)

    df = df.join(one_hot)

    return df





def one_hot(df, labels, drop_col=False):

    '''

    This function will one hot encode a list of columns.

    Args:

        df: Pandas dataframe

        labels: list of the columns to encode

        drop_col: boolean to decide if the chosen column should be dropped

    Returns:

        pandas dataframe with the given encoding

    '''

    for label in labels:

        df = one_hot_column(df, label, drop_col)

    return df
cabin_only = one_hot(cabin_only, ["Deck"],drop_col=True)
cabin_only.head()
for column in cabin_only.columns.values[1:]:

    mir_data[column] = cabin_only[column]
mir_data.drop(["Ticket","Cabin"], axis=1, inplace=True)
mir_data.head()
mir_data['Age'].isnull().sum()
train = mir_data

age_nan_rows = train[train['Age'].isnull()]
age_nan_rows.head()
train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

titles = train['Name'].unique()

titles
train['Age'].fillna(-1, inplace=True)
medians = dict()

for title in titles:

    median = train.Age[(train["Age"] != -1) & (train['Name'] == title)].median()

    medians[title] = median
for index, row in train.iterrows():

    if row['Age'] == -1:

        train.loc[index, 'Age'] = medians[row['Name']]
train.drop(["Name"], axis=1, inplace=True)

train.head()
train.isnull().sum()

#데이터 전처리
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import statsmodels.api as sm

import matplotlib.pyplot as plt

import itertools

import time
train.head()
from sklearn.linear_model import LinearRegression

X = train.drop(['Survived'], axis =1)

y = train[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y ,stratify = y,  random_state = 42)



lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))

print(lr.score(X_test, y_test))
logreg = LogisticRegression(C = 1000).fit(X_train, y_train)

print(logreg.score(X_train, y_train))

print(logreg.score(X_test, y_test))

