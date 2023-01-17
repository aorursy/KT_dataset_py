# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns



# Importation des données d'entrainement avec pandas

data = pd.read_csv('/kaggle/input/titanic/train.csv')

data.head(10)
sns.heatmap(data.isnull())
sns.countplot(x='Survived', hue = 'Pclass', data= data)
sns.countplot(x='Survived', hue = 'Sex', data= data)
sns.countplot(x='Survived', hue = 'Embarked', data = data)
sns.countplot(hue='Survived', x = 'Embarked', data = data)
sns.countplot(x='Pclass', hue = 'Embarked', data = data)
data['Age'].hist(bins = 4)
data['Fare'].hist(bins = 50, figsize = (10,5))
sns.boxplot(x='Pclass', y = 'Age', hue = 'Survived',data =data) # beaucoup mieux
sns.boxplot(x='Pclass', y = 'Fare', hue = 'Survived', data =data)
sns.boxplot(x='Survived', y = 'Fare', data =data)
# fonction de substitution

def rmpl_NaN_age(mat):

    age = mat[0]

    classe = mat[1]

    

    if pd.isnull(age):



        if classe == 1:

            return 38



        elif classe == 2:

            return 29



        else:

            return 22



    else:

        return age



data['Age'] = data[['Age','Pclass']].apply(rmpl_NaN_age,axis=1)

sns.heatmap(data.isnull())
data['Embarked'].fillna(value = 'S', inplace = True)

sx = pd.get_dummies(data['Sex'],drop_first=True)

embk = pd.get_dummies(data['Embarked'],drop_first=True)

sns.heatmap(data.isnull())
""" Suppression des colonnes inutiles et concatenation des dummies""" 

data.drop(['Cabin', 'Name', 'Ticket', 'Sex', 'Embarked'],axis=1,inplace=True) # suppression de la colonne Cabin

data = pd.concat([data,sx,embk],axis=1)

sns.heatmap(data.isnull())
""" Séparation des données en train et test sets """

from sklearn.model_selection import train_test_split



y = data['Survived']

data.drop('Survived',axis=1,inplace=True) 



X_train, X_test, y_train, y_test = train_test_split(data,y, test_size=0.3)
""" Création de notre modèle """

from sklearn.linear_model import LogisticRegression



model = LogisticRegression(solver = 'liblinear', verbose = 1)

model.fit(X_train,y_train)
#Prediction sur x_test

predict = model.predict(X_test)
# Evaluation du modèle en local

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test, predict))

print(accuracy_score(y_test, predict)) 
# application aux données test.csv

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sns.heatmap(test.isnull())
# Variables dummy test

sx2 = pd.get_dummies(test['Sex'],drop_first=True)

embk2 = pd.get_dummies(test['Embarked'],drop_first=True)
""" Suppression des colonnes inutiles et concatenation des dummies""" 

test.drop(['Cabin', 'Name', 'Ticket', 'Sex', 'Embarked'],axis=1,inplace=True) # suppression de la colonne Cabin

test = pd.concat([test,sx2,embk2],axis=1)

sns.heatmap(test.isnull())
test['Fare'].fillna(value = 7.75, inplace =True) # on remplace la donnée manquante par le prix 3eme classe standard

test['Age'] = test[['Age','Pclass']].apply(rmpl_NaN_age,axis=1)

sns.heatmap(test.isnull())

""" prediction à partir des données du fichier test"""

predict2 = model.predict(test)
""" Preparation du format de soumission """

soumission = pd.DataFrame({'PassengerId':test['PassengerId'],

                           'Survived':predict2})

soumission.head(10)
soumission.to_csv('my_Res.csv', index = False)
""" Visualisation des données de prédiction """

sns.countplot(predict2)
soumission['Survived'].value_counts()