# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

input_path = "../input/titanic/"

df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')

df_gender = pd.read_csv(input_path + 'gender_submission.csv')
# affichage des 10 primiers line et les derniers

df_test
# Affiche le survived par sex

sns.catplot(x='Sex', y='Survived', data=df_train, kind='bar');
# % de valeurs nulles pour chaque colonne

df_train.isna().sum()/len(df_train)*100
# verifie si pas de null pour les Age

df_train['Age'].isna().sum()

df_test['Age'].isna().sum()
#df_train['Age'] = df_train['Age'].fillna(0.0).astype(int)

#df_test['Age'] = df_test['Age'].fillna(0.0).astype(int)
# Trouver la moyenne des Age, puis arrondir la moyenne

train_moyen_age = int(df_train['Age'].mean())

test_moyen_age = int(df_test['Age'].mean())
# verification

train_moyen_age
# ramplacer les valeur null de l'Age par la moyenne 

df_train['Age'].fillna(train_moyen_age, inplace=True)

df_test['Age'].fillna(train_moyen_age, inplace=True)
# Verifie si pas de valeur null pour Fare

df_train['Fare'].isna().sum()

df_test['Fare'].isna().sum()
# Regroupement par Age df_train

df_train['Age_bins'] = pd.cut(df_train['Age'], 5, labels=np.arange(5))

df_train

# Regroupement par Age df_test

df_test['Age_bins'] = pd.cut(df_test['Age'], 5, labels=np.arange(5))

df_test
# Verification des donne df_train

df_train.describe()

# Verification des donne df_test

df_test.describe()
# ce graph vous permets de voir les Survived par tranche d'Age 

sns.catplot(y='Survived', x='Age_bins', data=df_train, kind='bar', height=5, aspect=3.4);
# NUM_FEATURES => variables numérique

# CAT_features => variables catégorielles

NUM_FEATURES = ['Parch', 'SibSp']

CAT_FEATURES = ['Age_bins', 'Sex']
# crée dataset avec les columns qui nous intéresse

X_train_NUM = df_train.loc[:, NUM_FEATURES]

X_train_cat = df_train.loc[:, CAT_FEATURES]
# on fusion le valeur numeraire et catégorielles

X_train = pd.concat([X_train_NUM, X_train_cat], axis=1)
# crée dataset avec les columns qui nous intéresse

X_test_NUM = df_test.loc[:, NUM_FEATURES]

X_test_cat = df_test.loc[:, CAT_FEATURES]
# on fusion le valeur numeraire et catégorielles

X_test = pd.concat([X_test_NUM, X_test_cat], axis=1)
X_train.info()
X_test.info()
y_train = df_train['Survived']
for c in X_train.select_dtypes('object').columns:

    X_train[c] = X_train[c].astype('category').cat.codes
for c in X_test.select_dtypes('object').columns:

    X_test[c] = X_test[c].astype('category').cat.codes
X_train.info()
X_test
X_test
#X_test = df_test.loc[:, ['Pclass', 'SibSp']]##
#df_predict = pd.DataFrame({'PassengerId'})
#df_predict.loc[df_test['Sex'] == 'female', 'predict'] = 1
#df_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print("train accuraccy: {}".format(train_acc))
X_test
#rf = pd.DataFrame()

#rf['predict'] = df_gender['Survived']

rf.predict(X_test)
#cellule pour stocker le résultat du modéle en appéllant son modéle rf et son set de test X_test

submission = pd.DataFrame({

    'PassengerId': df_test['PassengerId'],

    'Survived': rf.predict(X_test)

})



submission.to_csv('submission.csv', index=False)