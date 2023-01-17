# carregando bibliotecas

import pandas as pd
import numpy as np

import re
import math

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
# carregando bibliotecas

import pandas as pd
import numpy as np

import re
import math

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
# carregando dados
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

# guardando alguns dados que podem ser importantes no futuro
train_y = train["Survived"]
PassengerId = test['PassengerId']
train.head()
test.head()
train.info()
train = train.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
test = test.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)

train.head(3)
full_data = [train, test]

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

pd.crosstab(train['Title'], train['Sex']).plot(kind='bar', 
                                                    title ="Sexo por Title",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train = train.drop(["Name"], axis = 1)
test = test.drop(["Name"], axis = 1)

pd.crosstab(train['Title'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Title",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
train.head(3) # acompanhando o progresso
print("Valores únicos na coluna Pclass: " + str(np.sort(train.Pclass.unique())))
print(" ")
print( "Quantidade de pessoas na classe 1: " + str(train["Pclass"][train["Pclass"] == 1].count()))
print( "Quantidade de pessoas na classe 2: " + str(train["Pclass"][train["Pclass"] == 2].count()))
print( "Quantidade de pessoas na classe 3: " + str(train["Pclass"][train["Pclass"] == 3].count()))
print(" ")
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
train.head(3)
print("Valores únicos na coluna Sex: " + str(train.Sex.unique()))
print(" ")
print( "Quantidade total de mulheres: " + str(train["Sex"][train["Sex"] == "female"].count()))
print( "Quantidade total de homens: " + str(train["Sex"][train["Sex"] == "male"].count()))

pd.crosstab(train['Sex'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Sex",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
train.head(3)
print("Valores únicos na coluna SibSp: " + str(np.sort(train.SibSp.unique())))
pd.crosstab(train['SibSp'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por SibSp",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
print("Valores únicos na coluna Parch: " + str(np.sort(train.Parch.unique())))
pd.crosstab(train['Parch'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Parch",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
full_data = [train, test]

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por FamiliSize",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
pd.crosstab(train['IsAlone'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por IsAlone",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
train.head(3)
print("Valores únicos na coluna Embarked: " + str(train.Embarked.unique()))
print(" ")
print( "Quantidade de pessoa que embarcaram em S: " + str(train["Embarked"][train["Embarked"] == "S"].count()))
print( "Quantidade de pessoa que embarcaram em C: " + str(train["Embarked"][train["Embarked"] == "C"].count()))
print( "Quantidade de pessoa que embarcaram em Q: " + str(train["Embarked"][train["Embarked"] == "Q"].count()))
pd.crosstab(train['Embarked'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Embarked",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

data = pd.concat(( train, test ))

data = pd.get_dummies(data)
data = data.drop(['Embarked_S', 'Sex_male'], axis = 1)

train = data.iloc[:891,:]
test = data.iloc[891:,:]

train.head(3)
fig, ax_lst = plt.subplots(1, 2, figsize=(15, 10))
ax_lst[0].plot(train["Fare"][train["Survived"] == 1], "b.", train["Fare"][train["Survived"] == 0], "r.")
train.boxplot(column="Fare",by="Survived", ax = ax_lst[1])
fig.suptitle('Survived por Fare')
full_data = [train, test]

k = round(3.322 * math.log10(train.shape[0]) + 1) # fórmula de Sturges

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'], Fare_labels = pd.qcut(train['Fare'], k, retbins = True)
train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
for dataset in full_data:
    for i in range(len(Fare_labels)-1):
        dataset.loc[(dataset['Fare'] > Fare_labels[i]) & (dataset['Fare'] <= Fare_labels[i+1]), 'Fare'] = i

train = train.drop(['CategoricalFare'], axis = 1)

train.head(3)
fig, ax_lst = plt.subplots(1, 2, figsize=(15, 10))
ax_lst[0].plot(train["Age"][train["Survived"] == 1], "b.", train["Age"][train["Survived"] == 0], "r.")
train.boxplot(column="Age",by="Survived", ax = ax_lst[1])
fig.suptitle('Survived por Age')
full_data = [train, test]

for dataset in full_data:
    dataset['IsChild'] = dataset['Age'].apply(lambda x: 1 if x < 14 else 0)

pd.crosstab(train['IsChild'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por IsChild",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)
full_data = [train, test]

for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(train['Age'].mean())
train['CategoricalAge'], Age_labels = pd.qcut(train['Age'], k, retbins = True, duplicates='drop')
train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
for dataset in full_data:
    for i in range(len(Age_labels)-1):
        dataset.loc[(dataset['Age'] > Age_labels[i]) & (dataset['Age'] <= Age_labels[i+1]), 'Age'] = i

train = train.drop(['CategoricalAge'], axis = 1)

train.head(3)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show
train = train.drop(["SibSp", "Parch", "Title_Mr"], axis = 1)
test = test.drop(["SibSp", "Parch", "Title_Mr"], axis = 1)

train.head()
train_X = train.drop(["Survived"], axis = 1)
test_X = test.drop(["Survived"], axis = 1)

SScaler = StandardScaler()
SScaler  = SScaler.fit(train_X)

train_X = SScaler.transform(train_X)
test_X = SScaler.transform(test_X)
modelo = SVC()
modelo.fit(train_X, train_y) # Já tenho "train_y" desde o começo do código

predict = modelo.predict(test_X)
predict_dict = pd.DataFrame({'PassengerId': PassengerId, "Survived": predict})
predict_dict