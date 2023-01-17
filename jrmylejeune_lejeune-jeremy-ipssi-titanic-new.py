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
#get a list of the features within the dataset

#obtenir une liste des entités dans l'ensemble de données

print(df_train.columns)
df_train
df_train.describe()
df_train.info()
df_train.isna().sum() 
X_train = df_train.loc[:, ['Embarked', 'SibSp']]
y_train = df_train['Survived']
X_test = df_test.loc[:, ['Embarked', 'SibSp']]
#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=df_train)



#print percentages of females vs. males that survive

print("Pourcentage de femme qui ont survécu:", df_train["Survived"][df_train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Pourcentage d'homme qui ont survécu:", df_train["Survived"][df_train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=df_train)



#print percentage of people by Pclass that survived

print("Pourcentage de Pclass = 1 qui ont survécu:", df_train["Survived"][df_train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Pourcentage de Pclass = 2 qui ont survécu:", df_train["Survived"][df_train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Pourcentage de Pclass = 3 qui ont survécu:", df_train["Survived"][df_train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=df_train)



#I won't be printing individual percent values for all of these.

print("Pourcentage de SibSp = 0 qui ont survécu:", df_train["Survived"][df_train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Pourcentage de SibSp = 1 qui ont survécu:", df_train["Survived"][df_train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Pourcentage de SibSp = 2 qui ont survécu:", df_train["Survived"][df_train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=df_train)

plt.show()
#draw a bar plot of survival by Pclass

sns.catplot(x="Embarked", y="Survived", kind="bar", data=df_train);



print("Nombre de personne qui ont embarqué à Southampton (S):")

southampton = df_train[df_train["Embarked"] == "S"].shape[0]

print(southampton)



print("Nombre de personne qui ont embarqué à Cherbourg (C):")

cherbourg = df_train[df_train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Nombre de personne qui ont embarqué à Queenstown (Q):")

queenstown = df_train[df_train["Embarked"] == "Q"].shape[0]

print(queenstown)
#replacing the missing values in the Embarked feature with S

#remplacement des valeurs manquantes dans l'entité Embarked par S

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')
#conversion des valeurs en int

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)

df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)
#conversion des valeurs en int

embarked_mapping = {"female": 1, "male": 0}

df_train['Sex'] = df_train['Sex'].map(embarked_mapping)

df_test['Sex'] = df_test['Sex'].map(embarked_mapping)



df_train.head()
#Nombre de survivant total

df_train.Survived.value_counts()
df_train
df_train.isna().sum() 
df_train.sample(5)
#condition pour creation de la colonne AgeGroup

def define_agegroup(row):

    if row['Age'] < 0 :

        res='Unknown'

    elif row['Age'] <=5 :

        res='Baby'

    elif row['Age'] <=12 :

        res='Child'

    elif row['Age'] <=18 :

        res='Teenager'

    elif row['Age'] <=24 :

        res='Student'

    elif row['Age'] <=35 :

        res='Young Adult'

    elif row['Age'] <=60 :

        res='Adult'

    else :

        res='Senior'

    return res


df_train['AgeGroup'] = df_train.apply(define_agegroup, axis=1)

df_train

df_test['AgeGroup'] = df_test.apply(define_agegroup, axis=1)

df_test
#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=df_train)

plt.show()
plt.figure(figsize=(20, 7))



sns.distplot(df_train.loc[(df_train['Age'] > 50), 'Age']);
df_train['age_is_null'] = np.where(df_train['Age'].isnull() , 1, 0);

df_test['age_is_null'] = np.where(df_test['Age'].isnull() , 1, 0);
sns.catplot(x="age_is_null", y="Survived", kind="bar", data=df_train);
df_train.head()
#create a combined group of both datasets

combine = [df_train, df_test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])

pd.crosstab(df_test['Title'], df_test['Sex'])
#replace various titles with more common names

#remplacement / regoupement des titres/appelations

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



df_train
def define_fillage(row):

    if row['Title'] == 1 :

        res= 30

    elif row['Title'] == 2 :

        res= 19 

    elif row['Title'] == 3 :

        res= 25

    elif row['Title'] == 4 :

        res= 40

    elif row['Title'] == 5 :

        res= 45

    elif row['Title'] == 6 :

        res= 60

    else :

        res=70

    return res

df_train.loc[df_train['Age'].isna(),'Age'] = df_train.loc[df_train['Age'].isna()].apply(define_fillage, axis=1)

df_test.loc[df_test['Age'].isna(), 'Age'] = df_test.loc[df_test['Age'].isna()].apply(define_fillage, axis=1)

df_train
def define_fillagegroup(row):

    if row['Title'] == 1 :

        res='Adult'

    elif row['Title'] == 2 :

        res='Student'

    elif row['Title'] == 3 :

        res='Adult'

    elif row['Title'] == 4 :

        res='Adult'

    elif row['Title'] == 5 :

        res='Senior'

    elif row['Title'] == 6 :

        res='Young Adult'

    else :

        res='Adult'

    return res
df_train.loc[df_train['AgeGroup'].isna(), 'AgeGroup'] = df_train.loc[df_train['AgeGroup'].isna()].apply(define_fillagegroup, axis=1)

df_train



df_test.loc[df_test['AgeGroup'].isna(), 'AgeGroup'] = df_test.loc[df_test['AgeGroup'].isna()].apply(define_fillagegroup, axis=1)

df_test
#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

df_train['AgeGroup'] = df_train['AgeGroup'].map(age_mapping)

df_test['AgeGroup'] = df_test['AgeGroup'].map(age_mapping)
df_train
df_train['cabin_is_null'] = np.where(df_train['Cabin'].isnull() , 1, 0)

df_test['cabin_is_null'] = np.where(df_test['Cabin'].isnull() , 1, 0)
#draw a bar plot of Age vs. survival

sns.barplot(x="cabin_is_null", y="Survived", data=df_train)

plt.show()
sns.catplot(x="Survived", y="Fare", kind="bar", data=df_train);
sns.catplot(x="Survived", y="Fare", kind="box", data=df_train);
df_train = df_train.fillna({"Fare": 70})

df_test = df_test.fillna({"Fare": 70})
df_train['Fare'] = pd.cut(df_train['Fare'],4)

df_test['Fare'] = pd.cut(df_test['Fare'],4)

df_train
df_train['Fare'] = df_train['Fare'].astype('category').cat.codes

df_test['Fare'] = df_test['Fare'].astype('category').cat.codes

df_train
def define_riche_woman(row):

    if row['Fare'] <= 70 :

        res = 1

    else :

        res = 0

    return res
df_train['Rich_woman'] = df_train.apply(define_riche_woman, axis=1)

df_train

df_test['Rich_woman'] = df_test.apply(define_riche_woman, axis=1)

df_test
#draw a bar plot of Age vs. survival

sns.barplot(x="Rich_woman", y="Survived", data=df_train)

plt.show()
sns.catplot(x="Survived", y="Rich_woman", kind="box", data=df_train);
df_train = df_train.fillna({"Cabin": 0})

df_test = df_test.fillna({"Cabin": 0})
df_train['Age'] = pd.cut(df_train['Age'],5)

df_test['Age'] = pd.cut(df_test['Age'],5)
df_train['Age'] = df_train['Age'].astype('category').cat.codes

df_test['Age'] = df_test['Age'].astype('category').cat.codes

df_train
df_test.isna().sum() 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
#my_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'AgeGroup', 'Fare', 'cabin_is_null', 'Title']

#my_cols = ['Pclass', 'SibSp', 'Embarked', 'Sex', 'Title', 'cabin_is_null', 'AgeGroup', 'Fare', ]

my_cols = ['Pclass', 'SibSp', 'Embarked', 'Sex', 'Title', 'cabin_is_null', 'AgeGroup', 'age_is_null', 'Fare', 'Rich_woman', 'Age']
X_train = df_train.loc[:, my_cols]
X_train
y_train = df_train['Survived']
X_test = df_test.loc[:, my_cols]
X_train.info()
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



rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)