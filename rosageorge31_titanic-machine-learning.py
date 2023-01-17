import numpy as np

import pandas as pd
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



to_test=data.Survived

data.drop('Survived',1,inplace=True)

df=data.append(test)

df.reset_index(inplace=True)

df.drop('index',inplace=True,axis=1)
df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
grouped = df.groupby(['Sex','Pclass','Title'])





grouped.median()



df["Age"] = df.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
df.head()
df["Fare"] = df.groupby(['Sex','Pclass','Title'])['Fare'].transform(lambda x: x.fillna(x.median()))
df.head()
df['Sex'] = df['Sex'].map({'male':1,'female':0})

df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

df['Alone'] = df['FamilySize'].map(lambda s : 1 if s == 1 else 0)

df['Couple'] = df['FamilySize'].map(lambda s : 1 if s==2 else 0)

df['Family'] = df['FamilySize'].map(lambda s : 1 if 3<=s else 0)

df.Embarked.fillna('S',inplace=True)

df.Cabin.fillna('U',inplace=True)

df['Cabin'] = df['Cabin'].map(lambda c : c[0])

df.drop('Name',axis=1,inplace=True)
class_feature = pd.get_dummies(df['Pclass'],prefix="Pclass")

titles_feature = pd.get_dummies(df['Title'],prefix='Title')

embarked_feature = pd.get_dummies(df['Embarked'],prefix='Embarked')

cabin_feature = pd.get_dummies(df['Cabin'],prefix='Cabin')

df = pd.concat([df,cabin_feature],axis=1)

df = pd.concat([df,class_feature],axis=1)

df = pd.concat([df,titles_feature],axis=1)

df = pd.concat([df,embarked_feature],axis=1)

df.drop('Ticket',inplace=True,axis=1)

df.drop('Pclass',inplace=True,axis=1)

df.drop('Title',inplace=True,axis=1)

df.drop('Cabin',inplace=True,axis=1)





df.drop('Embarked',inplace=True,axis=1)



data = df.ix[0:890]

test = df.ix[891:1308]
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(data, to_test)

features = pd.DataFrame()

features['feature'] = data.columns

features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)