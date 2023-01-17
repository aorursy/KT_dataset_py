import pandas as pd

import numpy as np

import re

import os

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full_data = [train,test]
#Supervivencia segun su clase

print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
#Supervivencia segun el sexo

print(train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean())
#Creamos un nuevo feature sumando los features de pareja y padres/hijos

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
#Creamos un nuevo feature utilizando el anterior para saber si estaba solo

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg    = dataset['Age'].mean()

    age_std    = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
#Agrupamos en rare algunos titulos

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, col='Survived', row='Embarked', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, col='Survived', row='Parch', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train);
sns.barplot(x="Embarked", y="Survived", hue="Parch", data=train);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);
sns.pointplot(x="Embarked", y="Survived", hue="Pclass", data=train)
for dataset in full_data:

    

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

  

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

   

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

   

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] ;
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



#Ya tenemos todos nuestro datos de forma numerica
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['Feature']), 

           pd.DataFrame(rf.feature_importances_, columns = ['Importance'])), 

          axis = 1).sort_values(by='Importance', ascending = False)[:20]



#Importancia de las Features en nuestro Random Forest
predictions = rf.predict(test)

predictions = pd.DataFrame(predictions, columns=['Survived'])

predictions = pd.concat((test, predictions), axis = 1)

predictions



#Predicciones de nuestro RF