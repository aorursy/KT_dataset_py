# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
df_train.info()
df_train
df_train.isnull().sum()
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);

df_train[['Sex', 'Survived' ]].groupby(['Sex']).mean().sort_values(by='Survived')
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df_train, height=4, aspect=2);

df_train[['Pclass', 'Survived' ]].groupby(['Pclass']).mean().sort_values(by='Survived')
df_train["Hypothesis"] = 0

df_train.loc[df_train.Sex == "female", "Hypothesis"] = 1



df_train["Result"] = 0

df_train.loc[df_train.Survived == df_train["Hypothesis"], "Result"] = 1



# Fonction to remplace values for train and test data

def data_process(data):

    data = data.drop(['Name'], axis=1)



    data.loc[data["Sex"] == "male", "Sex"] = 0

    data.loc[data["Sex"] == "female", "Sex"] = 1



    # set age

    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[ data['Age'] <= 16, 'Age'] = int(0)

    data.loc[ data['Age'] <= 16, 'Age'] = int(0)

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age']



    # set Embarked

    data['Embarked'] = data['Embarked'].fillna(data.Embarked.dropna().mode()[0])

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



    # set Cabin

    data.drop(['Cabin'], axis=1)

    

    data = data.drop(['Fare'], axis=1)

    data = data.drop(['Ticket'], axis=1)

    data = data.drop(['Cabin'], axis=1)

    

    return data
df_train =  data_process(df_train)

df_train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(n_estimators=100)
y_train = df_train["Survived"]

X_train = df_train.drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

X_train = X_train.drop("Result", axis=1)



X_test_dataset = data_process(df_test)





#predicted_value = rf.predict(X_test_dataset)

#X_train.head()

rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
X_test = data_process(df_test)

rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)