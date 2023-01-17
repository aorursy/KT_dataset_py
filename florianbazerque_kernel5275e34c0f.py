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
df_train.info()
df_train.describe()
#Ici on peut voir que les jeunes personnes et les personnes plus agées ont beaucoup plus survécu que les personnes étant dans la trentaine 

sns.catplot(x="Age", y="Survived", kind="bar", data=df_train);
#Ici nous pouvons voir que les personnes ayant dépensé plus d'argent(donc plus riche ?) ont de manières générales mieux survécu à la catastrophe

sns.catplot(x="Fare", y="Survived", kind="bar", data=df_train);
#Sur ce graphe nous voyons que les gens étant en 1ere classe ont survécu à plus de 60% tandis que ceux en 3eme classe ont survécu à seulement ~25%

sns.catplot(x="Pclass", y="Survived", kind="bar", data=df_train);
#Sur ce graphe nous voyons que les personnes étant en 1ere classe ont une moyenne d'age bien plus élevée ce qui explique également pourquoi plus de personnes "agées" ont survécu (voir 1er graphe)

sns.catplot(x="Pclass", y="Age", kind="bar", data=df_train);
#On regarde combien de valeur sont NULL par colonne dans le tableau

df_train.isnull().sum()
#Séparation des âges NaN et âges remplis

df_train_no_age = df_train.loc[df_train['Age'].isnull()]

df_train_age = df_train.loc[df_train['Age'] > 0]
#On récupère les moyennes des âges par classe

mean_age_by_Pclass = df_train_age.groupby(['Pclass']).Age.mean()



#On met des index aux valeurs pour retrouver plus facilement la classe

index_ = [1, 2, 3]

mean_age_by_Pclass.index = index_



#On insère les valeurs dans le tableau

for age in mean_age_by_Pclass.iteritems(): 

    df_train.loc[(df_train['Pclass'] == age[0]) & (df_train['Age'].isnull()), 'Age'] = age[1]



#on vérifie que l'age a bien été inséré    

df_train.isnull().sum()
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 0

df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 1
#On effectue les opérations ci-dessus sur le df_test

#Séparation des âges NaN et âges remplis

df_test_no_age = df_test.loc[df_test['Age'].isnull()]

df_test_age = df_test.loc[df_test['Age'] > 0]
#On récupère les moyennes des âges par classe

mean_age_by_Pclass = df_test_age.groupby(['Pclass']).Age.mean()



#On met des index aux valeurs pour retrouver plus facilement la classe

index_ = [1, 2, 3]

mean_age_by_Pclass.index = index_



#On insère les valeurs dans le tableau

for age in mean_age_by_Pclass.iteritems(): 

    df_test.loc[(df_test['Pclass'] == age[0]) & (df_test['Age'].isnull()), 'Age'] = age[1]



#on vérifie que l'age a bien été inséré    

df_test.isnull().sum()
#Nous pouvons voir qu'il y a un champ Fare non rempli or nous en vavons besoin pour le predict

df_test.loc[df_test['Fare'].isnull()]
#Grace à la ligne du dessus nous savons quel ligne n'est pas remplie, nous allons donc faire une moyenne du prix payé en fonction de sa classe

fare = df_test.loc[df_test['Pclass'] == 3].groupby('Pclass').Fare.mean()

df_test.loc[df_test['Fare'].isnull(), 'Fare'] = fare.values[0]

#On vérifie que cela a bien été pris en compte

df_test.loc[df_test['Fare'].isnull()]
df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 0

df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 1
my_col = ['Age','Sex','Pclass','Fare']
X_train = df_train.loc[:, my_col]
y_train = df_train['Survived']
X_test = df_test.loc[:, my_col]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)