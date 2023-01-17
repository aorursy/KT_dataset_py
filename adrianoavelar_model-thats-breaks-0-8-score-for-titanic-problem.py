#modelos



# classifier models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# modules to handle data

import pandas as pd

import numpy as np
# load data 

train = pd.read_csv('../input/train.csv') 

test = pd.read_csv('../input/test.csv')

# Salvar PassengerId para a submissão final

passengerId = test.PassengerId



train = train.drop('PassengerId', axis = 1)

test = test.drop('PassengerId', axis = 1)

train.describe()
train.head()
train['Name'].describe()
train['Title'] = train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

train.head()

train['Title'].value_counts()
norm_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}



train.Title = train.Title.map(norm_titles)



train.Title.value_counts()
#Verificando a Média de Idade Agrupando Sex e Title.

train_grouped = train.groupby(['Sex','Title'])

train_grouped.Age.mean()
#Verificando a Mediana de Idade Agrupando Sex e Title.

train_grouped.Age.median()
#Aplicando os valores da média do grupo Sex/Title nos valores vazios de Age.

train.Age = train_grouped.Age.apply(lambda x: x.fillna(x.mean()))
#Checando quantos valores nulos existem em Age (Valor Esperado = 0)

train.Age.isnull().sum()
test['Title'] = test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

test.Title = test.Title.map(norm_titles)

test_grouped = test.groupby(['Sex','Title'])

test.Age = test_grouped.Age.apply(lambda x: x.fillna(x.mean()))

test.Age.isnull().sum()
# fill Cabin NaN with U for unknown

train.Cabin = train.Cabin.fillna('U')

# find most frequent Embarked value and store in variable

most_embarked = train.Embarked.value_counts().index[0]



# fill NaN with most_embarked value

train.Embarked = train.Embarked.fillna(most_embarked)

# fill NaN with median fare

train.Fare = train.Fare.fillna(train.Fare.median())



# view changes

train.info()
#O Máximo que dá para fazer com a coluna Cabin por hora é isolar a primeira letra e agrupá-la

#train['Cabin'] = train.Cabin.apply(lambda name: name[0])

train.Cabin = train.Cabin.map(lambda x: x[0])

#Misteriosamente na coluna train aparece um T e na coluna test não. Isso causa problemas,

#por isso precisei tirar o T e colocar outras letras.

train['Cabin'] = train.Cabin.replace({'T': 'G'})

train.Cabin.value_counts()
#Fazendo o mesmo para o conjunto de test



# fill Cabin NaN with U for unknown

test.Cabin = test.Cabin.fillna('U')

# find most frequent Embarked value and store in variable

most_embarked = test.Embarked.value_counts().index[0]



# fill NaN with most_embarked value

test.Embarked = test.Embarked.fillna(most_embarked)

# fill NaN with median fare

test.Fare = test.Fare.fillna(train.Fare.median())



#O Máximo que dá para fazer com a coluna Cabin por hora é isolar a primeira letra e agrupá-la

test['Cabin'] = test.Cabin.apply(lambda name: name[0])

test.Cabin.value_counts()
train['FamilySize'] = train.Parch + train.SibSp + 1

train['FamilySize'].describe()
#Same for Test



test['FamilySize'] = test.Parch + test.SibSp + 1

test['FamilySize'].describe()
# TRAIN

train.Sex = train.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features

pclass_dummies = pd.get_dummies(train.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(train.Title, prefix="Title")

cabin_dummies = pd.get_dummies(train.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(train.Embarked, prefix="Embarked")

# concatenate dummy columns with main dataset

train_dummies = pd.concat([train, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)



# drop categorical fields

train_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



train_dummies.head()
# TESTE

test.Sex = test.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features

pclass_dummies = pd.get_dummies(test.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(test.Title, prefix="Title")

cabin_dummies = pd.get_dummies(test.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(test.Embarked, prefix="Embarked")

# concatenate dummy columns with main dataset

test_dummies = pd.concat([test, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)



# drop categorical fields

test_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



test_dummies.head()
X = train_dummies.drop('Survived', axis=1).values 

y = train_dummies.Survived.values



# create param grid object 

forrest_params = dict(     

    max_depth = [n for n in range(9, 14)],     

    min_samples_split = [n for n in range(4, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(10, 60, 10)],

)
# instantiate Random Forest model

forrest = RandomForestClassifier()
# build and fit model 

forest_cv = GridSearchCV(estimator=forrest,param_grid=forrest_params, cv=5) 

forest_cv.fit(X, y)
print("Best score: {}".format(forest_cv.best_score_))

print("Optimal params: {}".format(forest_cv.best_estimator_))
test_dummies.head()
# create array for test set



forrest_pred = forest_cv.predict(test_dummies)

# dataframe with predictions

kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': forrest_pred})

# save to csv

kaggle.to_csv('titanic_pred.csv', index=False)