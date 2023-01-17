# importando bibliotecas que serão utilizadas

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.dtypes
(train.isnull().sum()/train.shape[0]).sort_values(ascending=False)
# Estatísticas gerais

train.describe()
# Incluindo variáveis não numericas

train.describe(include=['O'])
# Distribuição da varíaveis

train.hist(figsize=(10, 8))
# Analisando a taxa de sobrevivência nos grupos 'Pclass', 'Sex' e 'Embarked' 

sns.catplot(x='Pclass', y='Survived', data=train, kind='bar')

sns.catplot(x='Sex', y='Survived', data=train, kind='bar')

sns.catplot(x='Embarked', y='Survived', data=train, kind='bar')
# Média de sobreviventes de acordo pelo Sexo

train[['Sex', 'Survived']].groupby('Sex').mean()
# Analisando novamente o porto de embarque

sns.barplot(x="Sex", y="Survived", hue="Embarked", data=train)
# Impactio da idade na chance de sobrevivência

graficos = sns.FacetGrid(train, col="Survived", row='Sex')

graficos = graficos.map(plt.hist, 'Age')
grafico = sns.FacetGrid(train, col="Embarked", row='Survived')

grafico = grafico.map(plt.hist, 'Pclass')
# Salvando os índices

train_idx = train.shape[0]

test_idx = test.shape[0]



# Salvando o passengerId para a submissão no Kaggle

passenger_Id = test['PassengerId']



# Criando copia da coluna 'Survived para extrai-la durante a preparação'

target = train['Survived'].copy()

train.drop(['Survived'], axis=1, inplace=True)



# Concatenando os DataFrames

df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_merged.describe()
df_merged.describe(include=['O'])
# idade

mediana_idade = df_merged['Age'].median()

df_merged['Age'].fillna(mediana_idade, inplace=True)



#tarifa

mediana_tarifa = df_merged['Fare'].median()

df_merged['Fare'].fillna(mediana_tarifa, inplace=True)



# Porto de embarque

porto_frequente = df_merged['Embarked'].sort_values()[0]

df_merged['Embarked'].fillna(porto_frequente, inplace=True)
df_merged.head()
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})



embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')

df_merged = pd.concat([df_merged, embarked_dummies], axis=1)

df_merged.drop(['Embarked'], axis=1, inplace=True)

df_merged.head()
train = df_merged.iloc[:train_idx]

test = df_merged.iloc[train_idx:]
# Importanto os modelos

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
# Criando e treinando o primeiro modelo

dt_model = DecisionTreeClassifier(random_state=1)

dt_model.fit(train, target)
# Fazendo as predições

y_pred_dt = dt_model.predict(test)



# Colocando no modelo pedido pelo Kaggle para submissão

submission = pd.DataFrame({

    "PassengerId": passenger_Id,

    "Survived": y_pred_dt

})



# Gerando o arquivo 

submission.to_csv('./submission_dt.csv', index=False)
# criando e treinando segundo modelo

rf_model = RandomForestClassifier()

rf_model.fit(train, target)
# Fazendo as predições

y_pred_rf = rf_model.predict(test)



# Colocando no modelo pedido pelo Kaggle para submissão

submission = pd.DataFrame({

    'PassengerId': passenger_Id,

    'Survived': y_pred_rf

})



# Gerando o arquivo 

submission.to_csv('./submission_rf.csv', index=False)
median_fare = train['Fare'].median()

alex = np.array([3, 0, 21, 0, 0, median_fare, 0, 0, 1]).reshape((1,-1))
rf_model.predict(alex)[0]