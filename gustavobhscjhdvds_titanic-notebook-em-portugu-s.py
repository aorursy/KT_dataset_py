# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math 

import seaborn as sns

import missingno

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

import itertools



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head(30)
train.describe()

test.head()
test.describe()
print("A base de treino possui:")

print(train.memory_usage().sum(),"Bytes ou ")

print(train.memory_usage().sum()/1024,"KBytes ou ")

print(train.memory_usage().sum()/1024/1024,"MBytes")
print("E a base de teste... :")

print(test.memory_usage().sum(),"Bytes ou ")

print(test.memory_usage().sum()/1024,"KBytes ou ")

print(test.memory_usage().sum()/1024/1024,"MBytes")
print("Informações acerca de Train")

print("_"*40)

train.info()
print("Informações acerca de Test")

print("_"*40)

test.info()
print("Dimensões de Train")

print("Linhas, Colunas")

print(train.shape)
print("Dimensões de Test")

print("Linhas, Colunas")

print(test.shape)
print("Essas são todas as variáveis presentes: ", list(train))

print("_"*142)

print("|"*142)

print("_"*142)

print("Feita uma pequena análise, percebe-se que algumas são mais promissoras que outras, então "

      +"separamos algumas para verificar possiveis conjutos")



cont = 0

while cont<8:

    cont = cont+1

    for i in itertools.combinations(list(train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]),cont):

        print(i)


variaveis = ["SexBinario", "SibSpSex","Pclass"]
def SexBinario(valor):

    if valor == "female":

        return 1

    else:

        return 0

    

def EmbarkedNumerico(valor):

    if valor == "S":

        return 1

    elif valor == "C":

        return 2

    else:

        return 3

    

def GrupoIdade(idade):

    if idade>=65 :

        return 3

    elif idade < 65 and idade >=18 :

        return 2

    else:

        return 1



def SibSpBinario(sibsp):

    if sibsp != 1:

        return 0

    else:

        return 1

    

def CategFare(valor):

    if valor <= 100:

        return 0

    elif valor >100:

        return 1

    

def AgeBinario(valor):

    if valor<7:

        return 1

    else:

        return 0

    

def MenorDe7Anos(valor):

    if valor<7:

        return 1

    else:

        return 0

def SibSpSex(c):

    if c.Sex == "female" and c.SibSp <4:

        return 1

    else:

        return 0

    

train["EmbarkedNumerico"] = train["Embarked"].map(EmbarkedNumerico)

train["SexBinario"] = train["Sex"].map(SexBinario)

train["GrupoAge"] = train["Age"].map(GrupoIdade)

train["SibSpBinario"] = train["SibSp"].map(SibSpBinario)

train["CategFare"] = train["Fare"].map(CategFare)

train["AgeBinario"] = train["Age"].map(AgeBinario)

train["MenorDe7Anos"] = train["Age"].map(MenorDe7Anos)

train["SibSpSex"] = train.apply(SibSpSex, axis=1)





vivos = train[train["Survived"] == 1]





test["EmbarkedNumerico"] = test["Embarked"].map(EmbarkedNumerico)

test["SexBinario"] = test["Sex"].map(SexBinario)

test["GrupoAge"] = test["Age"].map(GrupoIdade)

test["SibSpBinario"] = test["SibSp"].map(SibSpBinario)

test["CategFare"] = test["Fare"].map(CategFare)

test["AgeBinario"] = test["Age"].map(AgeBinario)

test["MenorDe7Anos"] = test["Age"].map(MenorDe7Anos)

test["SibSpSex"] = test.apply(SibSpSex, axis=1)

vivos
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="Pclass")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="Sex")
plt.figure(figsize=(16,16))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="Age")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="SibSp")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="Parch")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="CategFare")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "male"], hue="Survived",y="EmbarkedNumerico")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "female"], hue="Survived",y="Pclass")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "female"], hue="Survived",y="Sex")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "female"], hue="Survived",y="Age")
plt.figure(figsize=(16,8))

sns.countplot(data=train[train["Sex"] == "female"], hue="Survived",y="SibSp")
plt.figure(figsize=(16,8))

plt.title("Sobrevivencia por SibSpSex")

sns.countplot(data=train, hue="Survived",y="SibSpSex")
plt.figure(figsize=(16,8))

plt.title("Grupo de Idade por Local de Embarque")

sns.countplot(data=train, hue="GrupoAge",y="Embarked")
plt.figure(figsize=(16,8))

plt.title("Grupo de Survived por Pclass")

sns.countplot(data=train, hue="Survived",y="Pclass")
plt.figure(figsize=(16,8))

plt.title("Grupo de Survived por Embarked")

sns.countplot(data=train, hue="Survived",y="Embarked")
plt.figure(figsize=(16,8))

plt.title("Grupo de idade por Survived")

sns.countplot(data=train, hue="Survived",y="GrupoAge")
plt.figure(figsize=(16,8))

plt.title("Grupo de Parch por Survived")

sns.countplot(data=train, hue="Survived",y="Parch")
plt.figure(figsize=(16,8))

plt.title("Grupo de SibSp por Survived")

sns.countplot(data=train, hue="Survived",y="SibSp")
plt.figure(figsize=(16,16))

plt.title("Idade por Survived")

sns.countplot(data=train, hue="Survived",y="Age")
plt.figure(figsize=(16,8))

plt.title("Grupo de Survived por Sex")

sns.countplot(data=train, hue="Survived",y="Sex")
plt.figure(figsize=(16,8))

plt.title("Grupo de Survived por Embarked")

sns.countplot(data=train, hue="Survived",y="Pclass")
plt.figure(figsize=(16,8))

plt.title("Grupo de Survived por CategFare")

sns.countplot(data=train, hue="Survived",y="CategFare")
X = train[variaveis]

y = train["Survived"]

X_prev = test[variaveis]
X = X.fillna(-1)

X_prev = X_prev.fillna(-1)
titanic_model = RandomForestClassifier(n_estimators=100,n_jobs=1,random_state=0)
titanic_model.fit(X,y)
p = titanic_model.predict(X_prev)
p
sub = pd.Series(p, index=test['PassengerId'],name="Survived" )

sub.to_csv("/kaggle/working/modelo_titanic_29.csv")