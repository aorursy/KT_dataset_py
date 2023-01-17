# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
#Récupération fichier train.csv

input_path = '../input/'

df_train = pd.read_csv(input_path+'titanic/train.csv')



# Fichier test

df_test = pd.read_csv(input_path+'titanic/test.csv')



#fichier submission

df_predict = pd.read_csv(input_path+'titanic/gender_submission.csv')
#initialisation des variables

liste_info = {

    "nombre_total_personnes" : 0,

    "nombre_total_hommes" : 0,

    "nombre_total_femmes" : 0,

    "nombre_survivants" : 0,

    "nombre_morts" : 0,

    "nombre_hommes_survivants" : 0,

    "nombre_hommse_morts" : 0,

    "nombre_femmes_survivantes" : 0,

    "nombres_femmes_mortes" : 0,

    "pourcentage_hommes" : 0,

    "pourcentage_femmes" : 0,

    "pourcentage_survivants" : 0,

    "pourcentage_morts" : 0,

    "pourcentage_hommes_morts" : 0,

    "pourcentage_hommes_survivants" : 0,

    "pourcentage_femmes_survivantes" : 0,

    "pourcentage_femmes_mortes" : 0,

    "age_maximal" : 0,

    "age_minimal" : 0,

    "age_moyen" : 0,

}



#initialisation du nombre total de personnes

liste_info['nombre_total_personnes'] = df_train['PassengerId'].size



#initialisation du nombre total d'hommes

liste_info['nombre_total_hommes'] = df_train['Sex'].loc[df_train['Sex'] == "male"].size



#initialisation du nombre total de femmes

liste_info['nombre_total_femmes'] = df_train['Sex'].loc[df_train['Sex'] == "female"].size



#initialisation du pourcentage d'hommes

liste_info['pourcentage_hommes'] = format((df_train['Sex'].loc[df_train['Sex'] == "male"].size / liste_info['nombre_total_personnes'])*100,'.2f')



#initialisation du pourcentage de femmes

liste_info['pourcentage_femmes'] = format((df_train['Sex'].loc[df_train['Sex'] == "female"].size / liste_info['nombre_total_personnes'])*100,'.2f')



#initialisation du nombre total de survivants

liste_info['nombre_survivants'] = df_train['Sex'].loc[df_train['Survived'] == 1].size



#initialisation du nombre total de morts

liste_info['nombre_morts'] = df_train['Sex'].loc[df_train['Survived'] == 0].size



#initialisation du pourcentage de survivants

liste_info['pourcentage_survivants'] = format((df_train['Sex'].loc[df_train['Survived'] == 1].size / liste_info['nombre_total_personnes'])*100,'.2f')



#initialisation du pourcentage de morts

liste_info['pourcentage_morts'] = format((df_train['Sex'].loc[df_train['Survived'] == 0].size / liste_info['nombre_total_personnes'])*100,'.2f')



#initialisation du nombre total d'hommes survivants

liste_info['nombre_hommes_survivants'] = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == "male")].size



#initialisation du nombre total d'hommes morts

liste_info['nombre_hommse_morts'] = df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == "male")].size



#initialisation du pourcentage d'hommes morts

liste_info['pourcentage_hommes_morts'] = format((df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == 'male')].size / liste_info['nombre_total_hommes'])*100,'.2f')



#initialisation du pourcentage d'hommes survivants

liste_info['pourcentage_hommes_survivants'] = format((df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 'male')].size / liste_info['nombre_total_hommes'])*100,'.2f')



#initialisation du nombre total de femmes survivantes

liste_info['nombre_femmes_survivantes'] = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == "female")].size



#initialisation du nombre total de femmes mortes

liste_info['nombres_femmes_mortes'] = df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == "female")].size



#initialisation du pourcentage de femmes morts

liste_info['pourcentage_femmes_mortes'] = format((df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == 'female')].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation du pourcentage de femmes survivants

liste_info['pourcentage_femmes_survivantes'] = format((df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 'female')].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation de l'age maximale

liste_info['age_maximal'] = max(df_train['Age'])



#initialisation de l'age minimal

liste_info['age_minimal'] = min(df_train['Age'])



#initialisation de l'age moyen

liste_info['age_moyen'] = format(df_train['Age'].mean(),'.0f')



#initialisation du prix moyen

liste_info['prix_moyen'] = format(df_train['Fare'].mean(),'.0f')



df_train.loc[df_train["Age"].isnull(),"Age"] = liste_info['age_moyen'] 

df_test.loc[df_test["Age"].isnull(),"Age"] = liste_info['age_moyen'] 
df_train.loc[df_train["Fare"].isnull(),"Fare"] = liste_info['age_moyen'] 

df_test.loc[df_test["Fare"].isnull(),"Fare"] = liste_info['age_moyen'] 
liste_info
df_train
df_test
df_predict = pd.DataFrame({'PassengerId': df_test['PassengerId'],

                           'prediction': 0})



df_predict.loc[df_test["Sex"] == 'female', 'prediction'] = 1
X_train = df_train.loc[:,['Pclass', 'Fare', 'Age']]

y_train = df_train['Survived']

X_test = df_test.loc[:,['Pclass', 'Fare', 'Age']]

rf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)

rf.predict(X_test)

# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)