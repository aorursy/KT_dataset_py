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
df_train
df_train.isna().sum()
sns.catplot(x="Survived", y="Age", kind="box", data=df_train);
#moyenne age

age_moyenne = int((df_train.loc[df_train['Age'] != 0])['Age'].mean())

age_moyenne
df_train['Age'].fillna(age_moyenne, inplace=True)

df_test['Age'].fillna(age_moyenne, inplace=True)



df_train['Age'] = df_train['Age'].fillna(0.0).astype(int)

df_test['Age'] = df_test['Age'].fillna(0.0).astype(int)
df_train.isna().sum()
df_train
df_train['Pclass_cat'] = pd.cut(df_train['Pclass'], 3, labels=np.arange(3))

df_test['Pclass_cat'] = pd.cut(df_test['Pclass'], 3, labels=np.arange(3))



df_test_Pclass_undefine = df_test.loc[df_test['Pclass_cat'].isna() == 1]

df_test_Pclass_undefine['Pclass'] = df_test['Pclass'].mean()



df_test.loc[df_test['Pclass_cat'].isna() == 1] = df_test_Pclass_undefine



df_test['Pclass_cat'] = pd.cut(df_test['Pclass'], 3, labels=np.arange(3))

df_test.loc[df_test['Pclass_cat'].isna() == 1]
# d'aprés le graph les personnes dans la classe 0 ont plus de chance de survivre

sns.catplot(x="Pclass_cat", y="Survived", data=df_train, kind='bar');
df_train['Age_cat'] = pd.cut(df_train['Age'], 5, labels=np.arange(5))

df_test['Age_cat'] = pd.cut(df_test['Age'], 5, labels=np.arange(5))



df_test_age_undefine = df_test.loc[df_test['Age_cat'].isna() == 1]

df_test_age_undefine['Age'] = df_test['Age'].mean()



df_test.loc[df_test['Age_cat'].isna() == 1] = df_test_age_undefine



df_test['Age_cat'] = pd.cut(df_test['Age'], 5, labels=np.arange(5))

df_test.loc[df_test['Age_cat'].isna() == 1]
# d'aprés le graph les plus jeune ont plus de chance de survivre

sns.catplot(x="Age_cat", y="Survived", data=df_train, kind='bar');
df_train['Fare_cat'] = pd.cut(df_train['Fare'], 5, labels=np.arange(5))

df_test['Fare_cat'] = pd.cut(df_test['Fare'], 5, labels=np.arange(5))



df_test_fare_undefine = df_test.loc[df_test['Fare_cat'].isna() == 1]

df_test_fare_undefine['Fare'] = df_test['Fare'].mean()



df_test.loc[df_test['Fare_cat'].isna() == 1] = df_test_fare_undefine



df_test['Fare_cat'] = pd.cut(df_test['Fare'], 5, labels=np.arange(5))

df_test.loc[df_test['Fare_cat'].isna() == 1]
# d'aprés le graph les personnes ayant payé leur billet plus chére ont plus de chance de survivre

sns.catplot(x="Fare_cat", y="Survived", data=df_train, kind='bar');
FEATURES_N = ['Parch', 'SibSp']

FEATURES_C = ['Fare_cat','Sex', 'Age_cat', 'Pclass_cat']

X_trainN = df_train.loc[:, FEATURES_N]

X_trainC = df_train.loc[:, FEATURES_C]
X_train = pd.concat([X_trainN, X_trainC], axis=1)
X_testN = df_test.loc[:, FEATURES_N]

X_testC = df_test.loc[:, FEATURES_C]

X_test = pd.concat([X_testN, X_testC], axis=1)
y_train = df_train['Survived']
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
for c in X_train.select_dtypes('object').columns:

    X_train[c] = X_train[c].astype('category').cat.codes
for c in X_test.select_dtypes('object').columns:

    X_test[c] = X_test[c].astype('category').cat.codes
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)