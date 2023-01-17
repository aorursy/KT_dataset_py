import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedKFold

from sklearn.preprocessing import LabelEncoder
treinamento = pd.read_csv('../input/titanic/train.csv')

teste = pd.read_csv('../input/titanic/test.csv')

dados = pd.concat([treinamento, teste])
dados.head(5) # Permite visualizar as primeiras linhas do DataFrame, de acordo com o argumento.
treinamento['Survived'].value_counts(normalize=True)
plt.figure(figsize=(8, 6))

sns.countplot(x = treinamento['Survived'], palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('')

plt.xticks((0, 1), ['Mortos', 'Sobreviventes'])

plt.show()
print(dados['Pclass'].value_counts())
plt.figure(figsize=(8, 6))

sns.countplot(x='Pclass', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Classe')

plt.xticks((0, 1, 2), ['Primeira', 'Segunda', 'Terceira'])

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados['Titulo'] = dados['Name'].str.extract('([A-Za-z]+)\.', expand = False)
dados['Titulo'].value_counts()
dados['Titulo']  = dados['Titulo'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Ms', 'Lady', 

                                            'Capt', 'Don', 'Jonkheer', 'Countess', 'Mlle','Mme', 

                                            'Dona'], 'Outro')
plt.figure(figsize=(8, 6))

sns.countplot(x='Titulo', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Título')

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados['Titulo'] = LabelEncoder().fit_transform(dados['Titulo'])
dados['Titulo'].value_counts()
plt.figure(figsize=(8, 6))

sns.countplot(x='Sex', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('')

plt.xticks((0, 1), ['Homens', 'Mulheres'])

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
sex_map = {'female':1, 'male':0}

dados['Sexo'] = dados['Sex'].map(sex_map).astype(int)
plt.figure(figsize=(8, 6))

sns.distplot(dados['Age'], color='navy')

plt.ylabel('Passageiros')

plt.xlabel('Idade')

plt.show()
print('Passageiros sem o valor da idade:',dados['Age'].isnull().sum())
plt.figure(figsize=(10, 8))

sns.boxplot(x = dados['Pclass'], y = dados['Age'], hue = dados['Sex'])

plt.ylabel('Idade',size=12)

plt.xlabel('Classe',size=12)

plt.show()
dados.groupby(['Sex', 'Pclass']).median()['Age']
dados.loc[(dados['Sex'] == 'male')   & (dados['Pclass'] == 1) & (dados['Age'].isnull()), 'Age'] = 42

dados.loc[(dados['Sex'] == 'male')   & (dados['Pclass'] == 2) & (dados['Age'].isnull()), 'Age'] = 30

dados.loc[(dados['Sex'] == 'male')   & (dados['Pclass'] == 3) & (dados['Age'].isnull()), 'Age'] = 25

dados.loc[(dados['Sex'] == 'female') & (dados['Pclass'] == 1) & (dados['Age'].isnull()), 'Age'] = 36

dados.loc[(dados['Sex'] == 'female') & (dados['Pclass'] == 2) & (dados['Age'].isnull()), 'Age'] = 28

dados.loc[(dados['Sex'] == 'female') & (dados['Pclass'] == 3) & (dados['Age'].isnull()), 'Age'] = 22
dados['Idade'] = pd.qcut(dados['Age'], 10, duplicates='drop')
plt.figure(figsize=(12, 8))

sns.countplot(x='Idade', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Faixa de Idade')

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados['Idade'] = LabelEncoder().fit_transform(dados['Idade'])
dados['Familiares'] = dados['SibSp'] + dados['Parch']

dados['Familiares'].value_counts()
familia_map = {0:0, 

               1:1, 2:1, 3:1, 

               4:2, 5:2, 

               6:3, 7:3, 10:3}

dados['Familiares'] = dados['Familiares'].map(familia_map)
plt.figure(figsize=(8, 6))

sns.countplot(x='Familiares', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Familiares a bordo')

plt.xticks((0, 1, 2, 3), ['0', '1-3', '4-5', '6+'])

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados['Frequencia_Ticket'] = dados.groupby('Ticket')['Ticket'].transform('count')
plt.figure(figsize=(8, 6))

sns.countplot(x='Frequencia_Ticket', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Aparições do ticket')

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados.loc[dados['Fare'].isnull()]
dados.groupby(['Pclass', 'Sex']).Fare.median()[3][0]
dados['Fare'] = dados['Fare'].fillna(10.4896)
dados['Taxa'] = pd.qcut(dados['Fare'], 10)
plt.figure(figsize=(16, 8))

sns.countplot(x='Taxa', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Faixa de preço')

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados['Taxa'] = LabelEncoder().fit_transform(dados['Taxa'])
print('Passageiros sem cabine:', dados['Cabin'].isnull().sum())

dados['Cabin'] = dados['Cabin'].fillna('N')
dados['Cabine'] = dados['Cabin'].str.slice(start = 0, stop = 1)
dados['Cabine'].value_counts()
dados['Cabine'] = dados['Cabine'].replace(['A', 'B', 'C', 'T'], '0')

dados['Cabine'] = dados['Cabine'].replace(['D', 'E'], '1')

dados['Cabine'] = dados['Cabine'].replace(['F', 'G'], '2')

dados['Cabine'] = dados['Cabine'].replace(['N'], '3')
plt.figure(figsize=(8, 6))

sns.countplot(x='Cabine', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Nível da Cabine')

plt.xticks((0, 1, 2, 3), ['N', 'A, B, C e T', 'D e E', 'F e G'])

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
dados.loc[dados['Embarked'].isnull()]
dados['Embarked'] = dados['Embarked'].fillna('S')
porto_map = {'C':0, 'Q':1, 'S':2}

dados['Porto'] = dados['Embarked'].map(porto_map).astype(int)   
plt.figure(figsize=(8, 6))

sns.countplot(x='Porto', data = dados, hue='Survived', palette=['darkred', 'navy'])

plt.ylabel('Passageiros')

plt.xlabel('Porto de Embarque')

plt.xticks((0, 1, 2), ['Cherbourg', 'Queenstown', 'Southampton'])

plt.legend(['Mortos', 'Sobreviventes'])

plt.show()
features = pd.concat([pd.get_dummies(dados['Pclass'], prefix='Class', drop_first=True),

                      pd.get_dummies(dados['Titulo'], prefix='Titulo', drop_first=True),

                      pd.get_dummies(dados['Sexo'], prefix='Sexo', drop_first=True),

                      pd.get_dummies(dados['Idade'], prefix='Idade', drop_first=True),

                      pd.get_dummies(dados['Familiares'], prefix='Familiares', drop_first=True),

                      pd.get_dummies(dados['Frequencia_Ticket'], prefix='Ticket', drop_first=True),

                      pd.get_dummies(dados['Taxa'], prefix='Taxa', drop_first=True),

                      pd.get_dummies(dados['Cabine'], prefix='Cabine', drop_first=True),

                      pd.get_dummies(dados['Porto'], prefix='Porto', drop_first=True),

                     ], axis = 1)
modelo = LogisticRegression()
X = features[0:891]

X.head()
X_test = features[891:1309]

X_test.head()
y = treinamento['Survived']

y.head()
accuracy_list = []



kf = RepeatedKFold(n_splits = 5, n_repeats = 10)



for train_lines, valid_lines in kf.split(X):

    X_train, X_valid = X.iloc[train_lines], X.iloc[valid_lines]

    y_train, y_valid = y.iloc[train_lines], y.iloc[valid_lines]

    

    modelo.fit(X_train, y_train)

    

    p_valid = modelo.predict(X_valid)

    

    accuracy_list.append(np.mean(y_valid == p_valid))
plt.figure(figsize=(8, 6))

plt.hist(accuracy_list)

plt.xlabel('Acurácia')

plt.ylabel('Repetições')

plt.show()
print(f'Acurácia média do treinamento: {np.mean(accuracy_list):.4f}')

print(f'Desvio padrão da acurácia treinamento: {np.std(accuracy_list):.4f}')
p = modelo.predict(X_test)

print(p)
submission = pd.Series(p, index=teste['PassengerId'], name='Survived')

submission.head()
submission.to_csv("submission.csv", header=True)