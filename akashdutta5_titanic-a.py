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
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head(5)
df.isnull().sum()
import seaborn as sns

sns.boxplot('Fare', data=df, orient='v')
sns.boxplot(x='Sex', y='Age', hue='Survived', data=df)
sns.set_style('whitegrid')

sns.boxplot(x='Survived', y='Age', hue='Sex', data=df)
sns.scatterplot(x='Age', y='Fare', data=df)
sns.distplot(df['Fare'].dropna(), bins=8)
def fill_age(df):

    df.Age.fillna(29.0, inplace=True)

    return df 



def fill_Fare(df):

    df.Fare.fillna(14, inplace=True)

    return df



def fill_Embarked(df):

    df.Embarked.fillna('S', inplace=True)

    return df



def drop_features(df):

    df=df.drop(['Ticket', "Cabin", 'PassengerId','Name'], axis=1)

    return df



def encode_label(df):

    from sklearn.preprocessing import LabelEncoder

    label=LabelEncoder()

    df['Embarked']=label.fit_transform(df['Embarked'])

    df['Sex']=label.fit_transform(df['Sex'])

    return df

def encoding_features(df):

    df=fill_age(df)

    df=fill_Fare(df)

    df=fill_Embarked(df)

    df=drop_features(df)

    df=encode_label(df)

    return df



df=encoding_features(df)
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.1, random_state=1)
def data_splitting(df):

    x=df.drop(['Survived'], axis=1)

    y=df['Survived']

    return x, y



x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)



from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score)
log_learning=log_model.predict(x_train)

learning_score=accuracy_score(y_train, log_learning)

print(learning_score)
proba=log_model.predict_proba(x_test)

proba[5]
prediction[5]