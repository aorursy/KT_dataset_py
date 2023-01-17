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



df_train
#initialisation du dictionnaire contenant toute les variables dont je pourrais avoir besoin pour la visualisation

liste_info = {

    "nombre_total_personnes" : 0,

    "nombre_total_hommes" : 0,

    "nombre_total_femmes" : 0,

    "nombre_total_enfants" : 0,

    "nombre_survivants" : 0,

    "nombre_morts" : 0,

    "nombre_hommes_survivants" : 0,

    "nombre_hommes_morts" : 0,

    "nombre_femmes_survivantes" : 0,

    "nombres_femmes_mortes" : 0,

    "nombre_enfants_survivants" : 0,

    "nombre_enfants_morts" : 0,

    "pourcentage_hommes" : 0,

    "pourcentage_femmes" : 0,

    "pourcentage_enfants" :0,

    "pourcentage_survivants" : 0,

    "pourcentage_morts" : 0,

    "pourcentage_hommes_morts" : 0,

    "pourcentage_hommes_survivants" : 0,

    "pourcentage_femmes_survivantes" : 0,

    "pourcentage_femmes_mortes" : 0,

    "pourcentage_enfants_survivants" : 0,

    "pourcentage_enfants_morts" : 0,

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



#initialisation du nombre total d'enfants'

liste_info['nombre_total_enfants'] = df_train['Sex'].loc[df_train['Age'] <= 18].size



#initialisation du pourcentage d'hommes

liste_info['pourcentage_hommes'] = format(

    liste_info['nombre_total_hommes'] / liste_info['nombre_total_personnes'] *100,

    '.2f'

)



#initialisation du pourcentage de femmes

liste_info['pourcentage_femmes'] = format(

        liste_info['nombre_total_femmes'] / liste_info['nombre_total_personnes'] *100 

        ,'.2f'

    )



#initialisation du pourcentage d'enfants

liste_info['pourcentage_enfants'] = format(

        liste_info['nombre_total_enfants'] / liste_info['nombre_total_personnes']*100,

        '.2f'

    )



#initialisation du nombre total de survivants

liste_info['nombre_survivants'] = df_train['Sex'].loc[df_train['Survived'] == 1].size



#initialisation du nombre total de morts

liste_info['nombre_morts'] = df_train['Sex'].loc[df_train['Survived'] == 0].size



#initialisation du pourcentage de survivants

liste_info['pourcentage_survivants'] = format(

        liste_info['nombre_survivants'] / liste_info['nombre_total_personnes'] * 100

        ,'.2f'

    )



#initialisation du pourcentage de morts

liste_info['pourcentage_morts'] = format(

        liste_info['nombre_morts']  / liste_info['nombre_total_personnes'] * 100

        ,'.2f'

    )

#initialisation du nombre total d'hommes survivants

liste_info['nombre_hommes_survivants'] = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == "male")].size



#initialisation du nombre total d'hommes morts

liste_info['nombre_hommes_morts'] = df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == "male")].size



#initialisation du nombre total d'enfants survivants

liste_info['nombre_enfants_survivants'] = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Age'] <=18)].size



#initialisation du nombre total d'enfants morts

liste_info['nombre_enfants_morts'] = df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Age'] <=18)].size



#initialisation du pourcentage d'hommes morts

liste_info['pourcentage_hommes_morts'] = format((df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == 'male')].size / liste_info['nombre_total_hommes'])*100,'.2f')



#initialisation du pourcentage d'hommes survivants

liste_info['pourcentage_hommes_survivants'] = format((df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 'male')].size / liste_info['nombre_total_hommes'])*100,'.2f')



#initialisation du nombre total de femmes survivantes

liste_info['nombre_femmes_survivantes'] = df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == "female")].size



#initialisation du nombre total de femmes mortes

liste_info['nombres_femmes_mortes'] = df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == "female")].size



#initialisation du pourcentage de femmes mortes

liste_info['pourcentage_femmes_mortes'] = format((df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Sex'] == 'female')].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation du pourcentage de femmes survivantes

liste_info['pourcentage_femmes_survivantes'] = format((df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Sex'] == 'female')].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation du pourcentage d'enfants morts

liste_info['pourcentage_enfants_morts'] = format((df_train['Sex'].loc[(df_train['Survived'] == 0) & (df_train['Age'] <= 18)].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation du pourcentage d'enfants survivants

liste_info['pourcentage_enfants_survivants'] = format((df_train['Sex'].loc[(df_train['Survived'] == 1) & (df_train['Age'] <= 18)].size / liste_info['nombre_total_femmes'])*100,'.2f')



#initialisation de l'age maximale

liste_info['age_maximal'] = max(df_train['Age'])



#initialisation de l'age minimal

liste_info['age_minimal'] = min(df_train['Age'])



#initialisation de l'age moyen

liste_info['age_moyen'] = format(df_train['Age'].mean(),'.0f')

#pour df_train

df_train['isJeune'] = 0 #on créé une colonne vide

df_train.loc[df_train['Age'] <= 18,'isJeune'] = 1 

#pour chaque personne dont l'age est inférieur ou égal à 18, on lui met 1 dans isJeune pour indiquer que la personne est un enfant





#pour df_test

df_test['isJeune'] = 0

df_test.loc[df_test['Age'] <=18 ,'isJeune'] = 1
#pour df_train



df_train['isJeuneRicheetFemme'] = 0

df_train.loc[(df_train['Age'] <= 30) & (df_train['Pclass'] == 1) & (df_train['Sex'] == 0),'isJeuneRicheetFemme'] = 1



#pour df_test

df_test['isJeuneRicheetFemme'] = 0

df_test.loc[(df_test['Age'] <= 30) & (df_test['Pclass'] == 1) & (df_test['Sex'] == 0),'isJeuneRicheetFemme'] = 1

#pour df_train



df_train['isJeunePauvreetFemme'] = 0

df_train.loc[(df_train['Age'] <= 30) & (df_train['Pclass'] == 3) & (df_train['Sex'] == 0),'isJeunePauvreetFemme'] = 1



#pour df_test

df_test['isJeunePauvreetFemme'] = 0

df_test.loc[(df_test['Age'] <= 30) & (df_test['Pclass'] == 3) & (df_test['Sex'] == 0),'isJeunePauvreetFemme'] = 1
#pour df_train



df_train['isJeuneRicheetHomme'] = 0

df_train.loc[(df_train['Age'] <= 30) & (df_train['Pclass'] == 1) & (df_train['Sex'] == 1),'isJeuneRicheetHomme'] = 1



#pour df_test

df_test['isJeuneRicheetHomme'] = 0

df_test.loc[(df_test['Age'] <= 30) & (df_test['Pclass'] == 1) & (df_test['Sex'] == 1),'isJeuneRicheetHomme'] = 1



#pour df_train



df_train['isJeunePauvreetHomme'] = 0

df_train.loc[(df_train['Age'] <= 30) & (df_train['Pclass'] == 3) & (df_train['Sex'] == 1),'isJeunePauvreetHomme'] = 1



#pour df_test

df_test['isJeunePauvreetHomme'] = 0

df_test.loc[(df_test['Age'] <= 30) & (df_test['Pclass'] == 3) & (df_test['Sex'] == 1),'isJeunePauvreetHomme'] = 1
df_train.loc[df_train["Age"].isnull(),"Age"] = liste_info['age_moyen'] 

df_test.loc[df_test["Age"].isnull(),"Age"] = liste_info['age_moyen'] 
df_train['Embarked'] = df_train['Embarked'].map({'S':0,'C':1,'Q':2})

df_test['Embarked'] = df_test['Embarked'].map({'S':0,'C':1,'Q':2})



df_train['Sex'] = df_train['Sex'].map({'male':0,'female':1})

df_test['Sex'] = df_test['Sex'].map({'male':0,'female':1})
df_train['categoryAge'] = pd.cut(df_train['Age'].astype(int),[0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0])

df_test['categoryAge'] = pd.cut(df_test['Age'].astype(int),[0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0])
df_train['categoryFare'] = pd.cut(df_train['Age'].astype(int),10)

df_test['categoryFare'] = pd.cut(df_test['Age'].astype(int),10)
objects = ('Hommes','Femmes','Enfants')

y_pos = np.arange(3) # Retourne des valeurs espacées uniformément dans un intervalle donné

performance = [liste_info['nombre_total_hommes'],liste_info['nombre_total_femmes'],liste_info['nombre_total_enfants']]

plt.bar(y_pos, performance, align='center', color=['red', 'blue', 'green'])

plt.xticks(y_pos, objects)

plt.ylabel('Nombre')

plt.title('Nombre de femmes,d\'hommes et d\'enfants')

plt.show()

#j'ai utilisé math plotlib car je suis beaucoup plus à l'aise dessus et je peux faire des diagrammes qui correspondents plus à ce que je veux montrer notamment au niveau des camemberts
labels = 'Hommes', 'Femmes','Enfants'

sizes = [liste_info['pourcentage_hommes'], liste_info['pourcentage_femmes'],liste_info['pourcentage_enfants']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()

ax = sns.countplot(x='categoryAge',  data=df_train)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
sns.distplot(df_train['Age'].astype(int));
ax = sns.countplot(x="Survived", data=df_train)
labels = 'Morts', 'Survivants'

sizes = [liste_info['pourcentage_morts'], liste_info['pourcentage_survivants']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
labels = 'Femmes', 'Hommes',

sizes = [liste_info['pourcentage_femmes_survivantes'], liste_info['pourcentage_hommes_survivants']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
labels = 'Femmes', 'Hommes'

sizes = [liste_info['pourcentage_femmes_mortes'], liste_info['pourcentage_hommes_morts']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
objects = ('Morts','Survivants')

y_pos = np.arange(2)

performance = [liste_info['nombre_hommes_morts'],liste_info['nombre_hommes_survivants']]

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Nombre')

plt.title('Nombre d\'hommes survivants et de morts')

plt.show()



labels = 'Morts', 'Survivants'

sizes = [liste_info['pourcentage_hommes_morts'], liste_info['pourcentage_hommes_survivants']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()

objects = ('Survivants','Morts')

y_pos = np.arange(2)

performance = [liste_info['nombre_enfants_survivants'],liste_info['nombre_enfants_morts']]

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Nombre')

plt.title('Nombre d\'enfants survivants et de morts')

plt.show()
labels = 'Morts', 'Survivants'

sizes = [liste_info['pourcentage_enfants_morts'], liste_info['pourcentage_enfants_survivants']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
objects = ('Survivantes','Mortes')

y_pos = np.arange(2)

performance = [liste_info['nombre_femmes_survivantes'],liste_info['nombres_femmes_mortes']]

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Nombre')

plt.title('Nombre de femmes survivantse et de mortes')

plt.show()
labels = 'Mortes', 'Survivantes'

sizes = [liste_info['pourcentage_femmes_mortes'], liste_info['pourcentage_femmes_survivantes']]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()

X_train = df_train.loc[:,['Pclass', 'Fare','Age','SibSp','Sex','Parch','isJeune','isJeuneRicheetFemme','isJeuneRicheetHomme']]

y_train = df_train['Survived']

X_test = df_test.loc[:,['Pclass', 'Fare','Age','SibSp','Sex','Parch','isJeune','isJeuneRicheetFemme','isJeuneRicheetHomme']]



#Rectification du NAN dans la colonne Fare de X_test

X_test['Fare'].loc[X_test['Fare'].isnull()] = 0.0





rf.fit(X_train, y_train)



train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)



# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)
submission
submission.to_csv('submission.csv', index = False)