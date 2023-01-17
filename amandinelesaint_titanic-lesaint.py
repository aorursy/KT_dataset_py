# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
input_path = "../input/titanic/"



df_train = pd.read_csv(input_path + 'train.csv')
df_train
df_test = pd.read_csv(input_path + 'test.csv')
df_test
df_train.describe()
df_test.describe()
#verifier si une valeur est nul dans df_train

df_train.isna().sum()
#verifier si une valeur est nul dans df_test

df_test.isna().sum()
#on enleve les valeur NaN donc nul 

age = df_train.loc[df_train['Age'] > 0]

age
#Il y a 177 valeur nul dans l'age donc calculons la moyenne de df_train

age_moyen = np.sum(age['Age'])//len(age['Age'])

age_moyen
#On va remplacer les valeur nul "NaN" par la moyenne qu'on a trouvé

age_null = df_train.loc[df_train['Age'].isnull()]

age_null
df_train['Age'].fillna(age_moyen, inplace=True)

df_train
df_test['Age'].fillna(age_moyen, inplace=True)

df_test
#On verifie si les ages sont encore manquante dans train

df_train.isna().sum()
#On verifie si les ages sont encore manquante dans test

df_test.isna().sum()
sns.distplot(df_train.Age);
df_filtered = df_train[df_train['Survived'] == 1]

df_filtered
sns.distplot(df_filtered.Age);
#Methode pour remplacé le sex en boleen

CAT_FEATURES = ['Sex', 'Embarked']

for c in CAT_FEATURES:

    df_train[c] = df_train[c].astype('category').cat.codes



df_train
for c in CAT_FEATURES:

    df_test[c] = df_test[c].astype('category').cat.codes

df_test
sns.catplot(x="Sex", y="Survived",kind="bar", data=df_train);
#Nombre total de survivant

nbr_total_surv = df_train['Sex'].loc[df_train['Survived'] == 1].size

nbr_total_surv
#Nombre total de survivant d'homme

nbr_total_surv_h = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 1)].size

nbr_total_surv_h
#Nombre total de survivant de femme

nbr_total_surv_f = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 0)].size

nbr_total_surv_f
nbr_total_surv_f2 = df_train.loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 0)]

nbr_total_surv_f2
nbr_total_surv_f2['Pclass'].value_counts() 
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df_train);
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_train);
sns.barplot(x='Pclass', y='Age', hue='Sex', data=df_train);
df_train
df_train['Family'] = df_train['SibSp'] + df_train['Parch']

df_train
sns.catplot(x="Family", y="Survived", kind="bar", data=df_train);
sns.barplot(x='Family', y='Survived', hue='Sex', data=df_train);
sns.catplot(x="Survived", y="Age", kind="box", data=df_train);
sns.catplot(x="Survived", y="Fare", kind="box", data=df_train);
#Le montant du ticket a t'il un impact sur les survivants

df_train['Fare_bins'] = pd.cut(df_train["Fare"], 3)

df_train
sns.catplot(x='Fare_bins', y="Survived", kind="bar", data=df_train, aspect=3.5);
my_col =['Age', 'Sex', 'Embarked', 'Pclass']
X_train = df_train.loc[:, my_col]

#X_train = df_train[:,['Sex', 'Age']] ce qu'on veut mais utilise NUM_FEATURES comme DAYS 3
y_train = df_train['Survived'] #ne changera jamais
X_test = df_test.loc[:, my_col]
#parti pour la creation du modele

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
#cellule pour stocker le résultat du modele en appelant son modele rf et son set de test X_test

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'],

                          'Survived': rf.predict(X_test)})



submission.to_csv('submission.csv', index=False)