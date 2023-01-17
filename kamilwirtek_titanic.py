import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train.head()
df_train.info()
g = sns.PairGrid(df_train)

g.map(plt.scatter)
sns.heatmap(df_train.isnull())
df_train['Cabin_known'] = df_train['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
#Looking for data correlated to Age

plt.figure(figsize=(12,6))

sns.heatmap(df_train.corr(), cmap='RdBu', annot= True)
plt.figure(figsize=(10,5))

sns.boxplot(data=df_train, y='Age', x='Pclass')
def impute_age(x):

    Age = x[0]

    Pclass = x[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(df_train['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
df_train.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis=1)
df_train.head()
df_train.dropna(inplace=True)
sns.heatmap(df_test.isnull())
df_test['Cabin_known'] = df_test['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)

sex = pd.get_dummies(df_test['Sex'],drop_first=True)

embark = pd.get_dummies(df_test['Embarked'],drop_first=True)

df_test = pd.concat([df_test,sex,embark],axis=1)

df_test.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1,inplace=True)

df_test.fillna(value=0, inplace=True)
y_train = df_train['Survived']

X_train = df_train.drop('Survived', axis=1)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(df_test)
pred = pd.DataFrame(predictions, columns=['Survived'])
pred['PassengerId'] = df_test['PassengerId']
pred.set_index('PassengerId', inplace = True)
pred.to_csv('prediction.csv')
print(classification_report(df_gender['Survived'], predictions))