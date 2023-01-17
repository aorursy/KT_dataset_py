# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/titanic"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

result_df = test_df[['PassengerId']]
train_df.head()
test_df.head()
#train_df['Name'].apply(lambda x: x.split(', ')[1].split('.')[0]).unique()

#Check Major prefixes in train columns

major_prefix = train_df['Name'].apply(lambda x: x.split(', ')[1].split('.')[0]).value_counts()[:5]

major_prefix_lst = list(major_prefix.index)

print(major_prefix_lst)
test_df['Name'].apply(lambda x: x.split(', ')[1].split('.')[0]).value_counts()
'''arr_cabin = train_df['Cabin'].unique()

initial_lst = []

for i in range(1, len(arr_cabin)):

    initial_lst.append(arr_cabin[i][0])

print(initial_lst)

#print([cabin[0]for cabin in arr_cabin])'''
train_df['Survived'].value_counts()
train_df['Cabin'].isnull().sum() / len(train_df)
train_df['Survived'].plot.hist()
train_df.isnull().sum()
def get_missing_columns(df):

    mis_vals = df.isnull().sum()

    mis_vals_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_vals, mis_vals_percent], axis=1)

    mis_val_ren_table = mis_val_table.rename(columns = {0: 'number of missing values', 

                                                       1: '% of total values'})

    mis_val_ren_table = mis_val_ren_table[mis_val_ren_table.iloc[:, 1]!=0].sort_values('% of total values', ascending=False)

    

    return mis_val_ren_table
get_missing_columns(train_df)
#乗組員だったのかを区分するカラムを追加

def add_has_cabin_num(df):

    df['has Cabin Number'] = df['Cabin'].notnull().apply(lambda x: int(x))

add_has_cabin_num(train_df)

train_df.head()
add_has_cabin_num(test_df)
test_df.head()
#年齢による区分。一旦不要で。

'''

def add_age_category(df):

    age_category_lst = list()

    import math

    for i in range(len(train_df)):

        age = train_df.iloc[i,5]

        if math.isnan(age):

            age_category_lst.append('nan')

        elif age <5:

            age_category_lst.append('baby')

        elif age <10:

            age_category_lst.append('child')

        elif age < 20:

            age_category_lst.append('10s')

        elif age < 30:

            age_category_lst.append('20s')

        elif age < 40:

            age_category_lst.append('30s')

        elif age < 50:

            age_category_lst.append('40s')

        elif age < 60:

            age_category_lst.append('50s')

        else:

            age_category_lst.append('senior')

  

    df['age_category'] = age_category_lst

#train_df'''
'''add_age_category(train_df)

train_df.head()'''
#敬称情報を作る関数

def convert_prefix(x):

    prefix = x.split(', ')[1].split('.')[0] 

    if prefix not in major_prefix_lst:

        prefix = 'Others'

    return prefix
#敬称の情報を追加

def add_prefix(df):

    df['prefix'] = df['Name'].apply(convert_prefix)

add_prefix(train_df)

train_df.head()
add_prefix(test_df)

test_df.head()
#家族総数カラムの追加

train_df['family_size'] = train_df['SibSp'] + train_df['Parch']
train_df.head()
test_df['family_size'] = test_df['SibSp'] + test_df['Parch']

test_df.head()
train_df['Cabin'].unique()
def cre_cabin_ini(col):

    if pd.isnull(col):

        ini = 'NAN'

    else:

        ini = col[0]

    return ini
#train_df['Cabin_initial'] = train_df['Cabin'].apply(cre_cabin_ini)

#train_df.head()
#test_df['Cabin_initial'] = test_df['Cabin'].apply(cre_cabin_ini)

#test_df.head()
train_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True, axis=1)

train_df.head()
test_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True, axis=1)

test_df.head()
X_train = train_df[['Pclass','Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'has Cabin Number', 'prefix', 'family_size']]

y_train = train_df[['Survived']]
X_test = test_df[['Pclass','Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'has Cabin Number', 'prefix', 'family_size']]
#seabornのcount plotを見てみる。

sns.countplot(x='Pclass', data=train_df, hue='Survived')
sns.countplot(x='Sex', data=train_df, hue='Survived')
sns.countplot(x='Embarked', data=train_df, hue='Survived')
sns.countplot(x='has Cabin Number', data=train_df, hue='Survived')
train_df[train_df['Age'].notnull()]['Age']
g = sns.FacetGrid(train_df, hue='Survived', height=10)

g.map(sns.distplot, 'Fare', kde=False)

g.add_legend()
g = sns.FacetGrid(train_df, hue='Survived', height=5)

g.map(sns.distplot, 'Age')

g.add_legend()
sns.countplot(x='SibSp', data=train_df, hue='Survived')
sns.countplot(x='Parch', data=train_df, hue='Survived')
sns.countplot('family_size', data=train_df, hue='Survived')
from sklearn.preprocessing import LabelEncoder
for col in ['Sex', 'Embarked', 'prefix']:

    le = LabelEncoder()

    le.fit(X_train[col].fillna('NA'))

    

    X_train[col] = le.transform(X_train[col].fillna('NA'))

    X_test[col] = le.transform(X_test[col].fillna('NA'))
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
pred_label = np.where(pred > 0.5, 1, 0)
result_df['Survived'] = pred_label
result_df.to_csv('submit.csv', index=False)
#パラメーターを最適化しながら実施
