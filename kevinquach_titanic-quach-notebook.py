# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
input_path = '../input/titanic/'

df_virgin = pd.read_csv(input_path + 'train.csv')

df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')

sub = pd.read_csv(input_path + 'gender_submission.csv')
df_train.sample(10)
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
df_train.isna().sum()
df_train.info()
df_train.describe()
mean_age = df_train['Age'].mean()



df_train['AgeIsNull'] = np.where(df_train['Age'].isnull() , 1, 0)

df_test['AgeIsNull'] = np.where(df_test['Age'].isnull() , 1, 0)



df_train['Age'].fillna(mean_age, inplace = True)

df_test['Age'].fillna(mean_age, inplace = True)



df_train['Age'] = df_train['Age'].astype(int)

df_test['Age'] = df_test['Age'].astype(int)
# sns.countplot(x='Embarked', data=df_train)

df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S', inplace = True)

df_test['Embarked'].fillna('S', inplace = True)
df_train['InCabin'] = np.where(df_train['Cabin'].isnull() , 1, 0)

df_test['InCabin'] = np.where(df_test['Cabin'].isnull() , 1, 0)
df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



df_train['Sex'] = df_train['Sex'].astype('category').cat.codes

df_train['Embarked'] = df_train['Embarked'].astype('category').cat.codes
COLUMNS = ['Sex', 'Pclass', 'Parch', 'SibSp', 'Embarked', 'InCabin']
f, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 5), sharey="row", sharex="col")



for i, c in enumerate(COLUMNS):

    sns.catplot(x=c, y="Survived", data=df_train, kind='bar', ax=axs[0, i])

    sns.countplot(x=c, data=df_train, ax=axs[1, i])

    plt.close()



plt.tight_layout()

plt.show()
sns.swarmplot(x="Survived", y="Fare", data=df_train.loc[df_train['Fare'] < 200], size=2)

sns.boxplot(x="Survived", y="Fare", data=df_train.loc[df_train['Fare'] < 200])
sns.distplot(df_train['Age'])
f, axs = plt.subplots(nrows=1, ncols=6, figsize=(25, 7))



for i, c in enumerate(COLUMNS):

    sns.catplot(x=c, y='Age', kind='box', hue='Survived', data=df_train, ax=axs[i])

    plt.close()
sns.boxplot(x='Embarked', y='Fare', data=df_virgin.loc[df_virgin['Fare'] < 300])
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_train['AgeBand'] = pd.cut(df_train['Age'], 5)

df_test['AgeBand'] = pd.cut(df_test['Age'], 5)



df_train['AgeBand'] = df_train['AgeBand'].astype('category').cat.codes

df_test['AgeBand'] = df_test['AgeBand'].astype('category').cat.codes
NEW_COLUMNS = ['FamilySize', 'AgeBand']

f, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))





for i, c in enumerate(NEW_COLUMNS):

    sns.catplot(x=c, y="Survived", data=df_train, kind='bar', ax=axs[0, i])

    sns.countplot(x=c, data=df_train, ax=axs[1, i])

    sns.boxplot(x="Survived", y=c, data=df_train, ax=axs[2, i])

    plt.close()
df_train['Fare'] = pd.cut(df_train['Fare'], 4)

df_test['Fare'] = pd.cut(df_test['Fare'], 4)
df_train.groupby(['Fare']).count()
df_train['Fare'] = df_train['Fare'].astype('category').cat.codes

df_test['Fare'] = df_test['Fare'].astype('category').cat.codes
f, axs = plt.subplots(ncols=2, figsize=(15, 5))



sns.catplot(x='Fare', y="Survived", data=df_train, kind='bar', ax=axs[0])

sns.countplot(x='Fare', data=df_train, ax=axs[1])

plt.close()
import re



def get_title(name):

    # Regex to get title

    # explication du regex : resort la première expression dont le forme est 

    # [espace + une ou plusieur letre (majuscule ou minuscule) + un point] 

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



def categoriseTitle(df):

    df['Title'] = df['Name'].apply(get_title)

    

    # On regarde les différents type de titre

    # On constate que plusieurs valeurs peuvent être regrouper

    

    # df['Title'].value_counts()

    

    # on regroupe les titres qui sont semblable

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

    # on groupe tous les autres titres pour en faire un groupe conséquent

    df.loc[(df['Title'] !='Mr') & (df['Title'] != 'Miss') & (df['Title'] != 'Master') & (df['Title'] !='Mrs'), 'Title'] = 'Other'

    # les données inconnues sont remplacés par Mr étant la valeur la plus présente

    df['Title'] = df['Title'].fillna('Mr')

    df['Title'] = df['Title'].astype('category').cat.codes

    

categoriseTitle(df_train)

categoriseTitle(df_test)

f, axs = plt.subplots(ncols=2, figsize=(15, 5))



sns.catplot(x='Title', y="Survived", data=df_train, kind='bar', ax=axs[0])

sns.countplot(x='Title', data=df_train, ax=axs[1])

plt.close()
# df_train = df_train.drop(columns="Name")

# df_train = df_train.drop(columns="Ticket")

# df_train = df_train.drop(columns="Fare")

# df_train = df_train.drop(columns="Cabin")
df_train.sample(5)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
NUM_FEATURES = ['SibSp', 'Parch', 'FamilySize', 'AgeIsNull']

CAT_FEATURES = ['Pclass', 'Embarked', 'Sex', 'Fare', 'Title','AgeBand']



X = df_train.loc[:, NUM_FEATURES]

X_cat = df_train.loc[:, CAT_FEATURES]

X_train = pd.concat([X, X_cat], axis=1)

y_train = df_train['Survived']



X_train.sample(10)
from sklearn.model_selection import train_test_split

def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))

    test_acc = accuracy_score(y_test, model.predict(X_test))

    return {

        'train accuracy': train_acc,

        'test accuracy': test_acc

    }



print(train_model(rf, X_train, y_train))
X_test = df_test.loc[:, NUM_FEATURES + CAT_FEATURES]
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

train_acc
rf.predict(X_test)
submission = pd.DataFrame({

    'PassengerId': df_test['PassengerId'],

    'Survived': rf.predict(X_test)

})
submission.to_csv('submission.csv', index=False)