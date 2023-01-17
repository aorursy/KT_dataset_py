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
# Importação de Pacotes



import pandas as pd 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
train.drop(['Name','Ticket','Cabin'],axis=1, inplace=True)

train.head()

test.drop(['Name','Ticket','Cabin'],axis=1, inplace=True)

test.head()
# Convertendo as variáveis categórica em variáveis indicadoras (get_dummies)

# TRAIN 



train=pd.get_dummies(train)

train.head()
# Convertendo as variáveis categórica em variáveis indicadoras (get_dummies)

# TEST



test = pd.get_dummies(test)

test.head()
#Verificando os valores ausentes 

#TRAIN

print ("TRAIN")

print(train.isnull().sum().sort_values(ascending=False))



print ("\n\n")



#TEST

print ("TEST")

print(test.isnull().sum().sort_values(ascending=False))
#Preencha os valores de NA / NaN

#TRAIN

train["Age"].fillna(train['Age'].mean(),inplace=True)



#TEST

test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)

#Verificando os valores ausentes 

#TRAIN

print ("TRAIN")

print(train.isnull().sum().sort_values(ascending=False))



print ("\n\n")



#TEST

print ("TEST")

print(test.isnull().sum().sort_values(ascending=False))
X = train.drop('Survived',axis=1)

y = train['Survived']
tree = DecisionTreeClassifier(max_depth=3,random_state=0)

tree.fit(X,y)

tree.score(X,y)

score_tree = cross_val_score(tree, X, y, cv=5).mean()

print(score_tree)

submit = pd.DataFrame()

submit['PassengerId']= test['PassengerId']

submit['Survived']=tree.predict(test)





submit.to_csv("../working/submit.csv", index=False)
submit.head(10)
submit.shape