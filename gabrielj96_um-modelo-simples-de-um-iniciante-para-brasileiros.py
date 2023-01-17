import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
treino = pd.read_csv('../input/titanic/train.csv')

validacao = pd.read_csv('../input/titanic/test.csv')
treino.head()
validacao.head()
treino.info()

print('_'*40)

validacao.info()
treino["Embarked"].value_counts()
ax = sns.barplot(x = 'Embarked', y = 'Fare', data = treino)

ax
treino[treino['Embarked'].isna()]
treino["Embarked"] = treino["Embarked"].fillna("C")
novos_valores = {'S': 1, 'C': 2, 'Q': 3}

treino["Embarked"] = treino["Embarked"].map(novos_valores)

validacao["Embarked"] = validacao["Embarked"].map(novos_valores)
treino.head()
validacao.head()
treino.info()

print('_'*40)

validacao.info()
treino["Sex"] = pd.get_dummies(treino["Sex"])

validacao["Sex"] = pd.get_dummies(validacao["Sex"])
treino['Cabin'].unique()
treino['Cabin'].isna().sum()
variaveis_reservas = ["Cabin", "Ticket"]

treino = treino.drop(variaveis_reservas, axis = 1)

validacao = validacao.drop(variaveis_reservas, axis = 1)
pd.set_option('display.max_rows', None)

treino.sort_values('Survived')
treino_vivos = treino[treino["Survived"] == 0].copy()

treino_mortos = treino[treino["Survived"] == 1].copy()



media_age_vivos = treino_vivos["Age"].mean()

media_age_mortos = treino_mortos["Age"].mean()
treino.loc[treino['Survived'] == 0, 'Age'] = treino.loc[treino['Survived'] == 0, 'Age'].fillna(media_age_mortos)

treino.loc[treino['Survived'] == 1, 'Age'] = treino.loc[treino['Survived'] == 1, 'Age'].fillna(media_age_vivos)
validacao['Age'] = validacao['Age'].fillna(validacao['Age'].mean())
validacao[validacao['Fare'].isna()]
validacao['Fare'] = validacao['Fare'].fillna(validacao['Fare'].median())
treino.head()
validacao.head()
treino.corr()
ax = sns.pairplot(treino, y_vars='Survived', x_vars=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],

                  kind = 'reg')

ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)

ax
g = sns.FacetGrid(treino, col = 'Survived', height = 5)

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(treino, col = 'Survived', height = 5)

g.map(plt.hist, 'SibSp', bins=20)
g = sns.FacetGrid(treino, col = 'Survived', height = 5)

g.map(plt.hist, 'Parch', bins=20)
treino[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
treino[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Significado de cada variável



# Survived: Se sobreviveu ou não

# Pclass: Ticket de classe(Status socio-econômico)

# Name = Nome

# Sex: Sexo

# Age: Idade

# SibSp: Número de irmãos/irmãs ou cônjuges abordo

# Parch: Número de pais ou crianças abordo

# Ticket: Número do Ticket

# Fare: Tarifa paga

# Cabin: Número da cabine

# Embarked: Porta de embarque
treino.head()
treino['FamilySize'] = treino['SibSp'] + treino['Parch']

    

validacao['FamilySize'] = validacao['SibSp'] + validacao['Parch']
treino.head()
validacao.head()
treino[["FamilySize", "Survived"]].groupby(['FamilySize'],

                                           as_index=False).mean().sort_values(by='Survived', ascending=False)
treino = treino.drop("SibSp", axis=1)

treino = treino.drop("Parch", axis=1)



validacao = validacao.drop("SibSp", axis=1)

validacao = validacao.drop("Parch", axis=1)
treino['Title'] = treino['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])



validacao['Title'] = validacao['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
treino[["Title", "Survived"]].groupby(['Title'],

                                           as_index=False).mean().sort_values(by='Survived', ascending=False)
treino.head()
validacao.head()
treino = treino.drop("Name", axis=1)

validacao = validacao.drop("Name", axis=1)
titulos_treino = treino['Title'].unique()

total_titulos_treino = len(titulos_treino)

mapeamento_treino = dict(zip(titulos_treino, range(total_titulos_treino)))



titulos_val = validacao['Title'].unique()

total_titulos_val = len(titulos_val)

mapeamento_val = dict(zip(titulos_val, range(total_titulos_val)))
treino['Title'] = treino['Title'].replace(mapeamento_treino)

validacao['Title'] = validacao['Title'].replace(mapeamento_val)
treino.head()
validacao.head()
treino.info()
validacao.info()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
treino = treino.drop("PassengerId", axis=1)
X = treino.drop('Survived', axis = 1).copy()

Y = treino['Survived'].copy()

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size= 0.5, random_state= 1)
X_validacao  = validacao.drop("PassengerId", axis=1).copy()

X_treino.shape, X_teste.shape, Y_treino.shape, Y_teste.shape, X_validacao.shape
random_forest = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state = 1)

random_forest.fit(X_treino, Y_treino)

Y_predict = random_forest.predict(X_teste)

random_forest.score(X_teste, Y_teste)

acc_random_forest = round(random_forest.score(X_teste, Y_teste) * 100, 2)

acc_random_forest
Val_predict = random_forest.predict(X_validacao)
Sexto_modelo_feito_por_mim = pd.DataFrame({

        "PassengerId": validacao["PassengerId"],

        "Survived": Val_predict

    })
#Sexto_modelo_feito_por_mim.to_csv('Sexto_modelo_feito_por_mim.csv', index=False)
Primeiro_modelo_feito_por_mim_score = 0.74162
Segundo_modelo_feito_por_mim_score = 0.75119
Terceiro_modelo_feito_por_mim_score = 0.76076
Quarto_modelo_feito_por_mim_score = 0.76555
Quarto_modelo_feito_por_mim_score = 0.77990