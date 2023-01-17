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

df_train
my_col = ['Sex', 'Embarked']
#X_train = df_train.loc[:,['Pclass', 'Age']]

X_train = df_train.loc[:,my_col]
y_train = df_train['Survived']
df_test = pd.read_csv(input_path + 'test.csv')

df_test
df_train.sample(10)
df_test.sample(10)
X_test = df_test.loc[:,my_col]
# Verifie le nombre de champ null dans df_train 

df_train.isnull().sum()
# verifie le nombre de NaN dans le df_test

df_test.isnull().sum()
# recuperation des ages non NaN

age = df_train.loc[df_train['Age'] > 0]

age
# calcul de la moyen general de l'age des passager

age_moyen = np.sum(age.Age)//len(age.Age)

age_moyen
age_null = df_train.loc[df_train['Age'].isnull()]

age_null
age_null.loc[:,'Age'] = age_moyen

age_null
#Ajouet de tout les age Nan avec le fontion fillna(), l'age moyen a etait calculer plus haut 

df_train['Age'].fillna(age_moyen, inplace=True)

df_train
#Verification que les ages NaN son bien remplacer

df_train.isnull().sum()
df_test['Age'].fillna(age_moyen, inplace=True)

df_test
#Verification que les ages NaN son bien remplacer

df_test.isnull().sum()
# on recupere les Fare null de df test 

df_test.loc[df_test['Fare'].isnull()]
# on verifie si il existe dans train

df_train.loc[df_train['PassengerId'] == 1044]
#on vas recuper le fare moyen de train

fare = df_train.loc[df_train['Fare'] > 0]

fare_moyen = np.sum(fare.Fare)//len(fare.Fare)

fare_moyen
#on ajoute le fare moyen de train dans les valeur NaN de df_test

df_test['Fare'].fillna(fare_moyen, inplace=True)
df_test.loc[df_test['PassengerId'] == 1044]
# creation d'une colonne sex en boolean initialiser a 0

df_train['Sex_bool'] = 0

df_train
# ajout de la valeur 1 si son il est un homme

df_train['Sex_bool'].loc[df_train['Sex'] == 'male'] = 1

df_train
# meme chose que pour train 

df_test['Sex_bool'] = 0
df_test['Sex_bool'].loc[df_test['Sex'] == 'male'] = 1
# q_cut pour decouper l'age en plusieur partie(5), puis on les labels pour pouvoir les passes en paramettre dans l'entrainement 

df_train['age_cut'] = pd.qcut(df_train['Age'], 5, labels=[0, 1, 2, 3, 4])
df_train
df_test['age_cut'] = pd.qcut(df_test['Age'], 5, labels=[0, 1, 2, 3, 4])
df_test
df_train
# transformation de valeur string en numerique pour les embarked

# on regarde qu'elle est la valeur la plus rependu 

Embarked = df_train.loc[df_train['Embarked'].notnull()]

Embarked['Embarked'].describe()
# puis on l'ajoute a tout les valeur manquante 

df_train['Embarked'].fillna('S', inplace = True)
# on addition les deux colone pour recuper le nombre total de parsonne par famille 

df_train['Family'] = df_train['SibSp'] + df_train['Parch'] + 1

#df_train.tail()
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1

#df_test.tail()
df_train.loc[df_train['Cabin'].notnull()]
name = df_train['Name'].copy()
# je decoupe mes serie a partir de la ',' puis si je rencontre un carractere a à z et A à Z , je prend tous se qu'il y a apres jusqu'au point

name = name.str.extract(', ([a-zA-Z]+).')
df_train['Titel_name_extract'] = name
df_train['Titel_name_extract'].value_counts()
df_train['Titel_name_extract_map'] = df_train['Titel_name_extract'].map({'Mr' : 0,

'Miss' : 1,

'Mrs' : 2,

'Master' : 3

})
df_train['Titel_name_extract_map'].fillna(4, inplace=True)
df_train
name_test = df_test['Name'].copy()

name_test = name_test.str.extract(', ([a-zA-Z]+).')

df_test['Titel_name_extract'] = name_test

df_test['Titel_name_extract'].value_counts()
df_test['Titel_name_extract_map'] = df_test['Titel_name_extract'].map({'Mr' : 0,

'Miss' : 1,

'Mrs' : 2,

'Master' : 3

})
df_test['Titel_name_extract_map'].fillna(4, inplace=True)
df_test
cabin = df_train.copy()

#cabin
df_train['Cabin_bool'] = 0
df_train['Cabin_bool'].loc[df_train['Cabin'].notnull()] = 1
sns.catplot(x='Survived', y='Cabin_bool', kind='bar', data=df_train);
df_test['Cabin_bool'] = 0

df_test['Cabin_bool'].loc[df_test['Cabin'].notnull()] = 1
df_train['Ticket'].describe()
df_test['Ticket'].describe()
df_train.loc[df_train['Ticket'] == "CA. 2343"]
ticket = df_train['Ticket'].copy()
ticket = ticket.str.extract(' ([0-9]+)')
df_train['Ticket_extract'] = ticket

df_train['Ticket_extract'].value_counts()
df_train['Ticket_extract'].loc[df_train['Ticket_extract'].isnull()] = df_train.loc[df_train['Ticket_extract'].isnull()]['Ticket']
df_train.loc[:, ['Ticket', 'Ticket_extract']]
df_train.loc[178:, ['Ticket', 'Ticket_extract']]
df_train['Ticket_extract'].loc[df_train['Ticket_extract'] == 'LINE'] = -1
df_train['Ticket_extract'] = pd.to_numeric(df_train['Ticket_extract'])
df_train['Ticket_extract_cut'] = pd.qcut(df_train['Ticket_extract'], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
sns.catplot(x='Ticket_extract_cut', y='Survived', kind='bar', height=7, aspect=1.5, data=df_train);
ticket_test = df_test['Ticket'].copy()

ticket_test = ticket_test.str.extract(' ([0-9]+)')

df_test['Ticket_extract'] = ticket_test

df_test['Ticket_extract'].loc[df_test['Ticket_extract'].isnull()] = df_test.loc[df_test['Ticket_extract'].isnull()]['Ticket']

df_test['Ticket_extract'].loc[df_test['Ticket_extract'] == 'LINE'] = -1

df_test['Ticket_extract'] = pd.to_numeric(df_test['Ticket_extract'])

df_test['Ticket_extract_cut'] = pd.qcut(df_test['Ticket_extract'], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
sns.catplot(x='Ticket_extract_cut', y='Survived', kind='bar', height=7, aspect=1.5, data=df_train);
# visuelle d'impate en fonction du sexe (homme, femme)

sns.catplot(x='Sex', y='Survived', kind='bar', data=df_train);
sns.catplot(x='Pclass', y='Survived', kind='bar', data=df_train);
sns.catplot(x='Sex', y='Pclass', kind='bar', data=df_train);
sns.catplot(x='Sex', y='Age', kind='boxen', hue='Survived', data=df_train);
sns.catplot(x='age_cut', y='Survived', kind='bar', height=7, aspect=1.5, data=df_train);
sns.catplot(x="Pclass", y="Age", hue="Survived", kind="bar", height=7, aspect=2.5, data=df_train);
df_femme = df_train.loc[df_train['Sex'] == 'female' ]
sns.catplot(x='Pclass', y='Survived', kind="bar", data=df_train);
df_homme = df_train.loc[df_train['Sex'] == 'male' ]
sns.catplot(x='Pclass', y='Survived', kind="bar", data=df_homme);
sns.relplot(x='Age', y='Fare', hue='Survived', data=df_train);
# on decoupe en 5 par egale les prix des tickets 

df_train['Fare_cut'] = pd.qcut(df_train['Fare'], 4, labels=[0, 1, 2, 3])

df_test['Fare_cut'] = pd.qcut(df_test['Fare'], 4, labels=[0, 1, 2, 3])
sns.catplot(x='Fare_cut', y='Survived', hue="Sex", kind="bar", height=7, aspect=2.5, data=df_train);
sns.barplot(x='Embarked', y='Survived', data=df_train);
sns.barplot(x='Embarked', y='Fare', data=df_train);
# on convertie nos embarked en numerique pour pouvoir les exploites dans l'entrainement 

df_train['Embarked_map'] = df_train['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2})
df_test['Embarked_map'] = df_test['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2})
df_train[1:29]
# commenter pour gagner du temps lors d'un run all

sns.set(style="ticks")

col = ['Survived', 'age_cut', 'Sex_bool', 'Fare_cut', 'Family', 'Pclass', 'Embarked_map', 'Age']

#sns.pairplot(df_train.loc[:,col], hue="Survived");
sns.catplot(x='Titel_name_extract', y='Survived', kind="bar", height=7, aspect=2.5, data=df_train);
#sns.catplot(x='Titel_name_extract_map', y='Survived', kind="bar", height=7, aspect=2.5, data=df_train);

plt.figure()

ax = sns.barplot(x='Titel_name_extract_map', y='Survived', data=df_train);

ax.set_xticklabels(labels=['Mr', 'Miss', 'Mrs', 'Master' , 'Autre']);
sub = pd.read_csv(input_path + 'gender_submission.csv')
sub.columns
my_col = ['age_cut', 'Sex_bool', 'Fare_cut', 'Family', 'Pclass', 'Embarked_map', 'Titel_name_extract_map']

# le cabin_bool n'est pas assez pertinente donc on le retire
X_train = df_train.loc[:, my_col]
df_train
y_train = df_train['Survived']
X_train.isnull().sum()
X_test = df_test.loc[:, my_col]  
sns.catplot(x='Sex', y='Survived', kind="bar", data=df_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(X_train, y_train)
# y paramettre a predire , rf.predict() pour lancer le test 

train_acc = accuracy_score(y_train, rf.predict(X_train))

print("train accuraccy: {}".format(train_acc))
rf.predict(X_test)
X = df_test.loc[:,my_col]

fi_dict = {

    'feats': X.columns,

    'feature_importance': rf.feature_importances_

}

fi = pd.DataFrame(fi_dict).set_index('feats').sort_values(

    'feature_importance', ascending=False)

fi.sort_values(

    'feature_importance', ascending=True).tail(10).plot.barh();
#rf = pd.DataFrame()

#rf['predict'] = sub['Survived']
# cellule pour stocker le resultat du modele

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)
from sklearn.model_selection import train_test_split

def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))

    test_acc = accuracy_score(y_test, model.predict(X_test))

    return {

        'train accuracy': train_acc,

        'test accuracy': test_acc

    }

print(train_model(rf, X_train, y_train))