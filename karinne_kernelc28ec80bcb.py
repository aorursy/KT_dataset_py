# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn import model_selection

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline
# Lendo os conjuntos teste e treino

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = [train,test]
# Visualizando as 5 primeiras linhas do conjunto de treino.

train.head()
# Visualizando as 5 primeiras linhas do conjunto de teste.

test.head()
# Variáveis numéricas.

train.describe()
# Variáveis numéricas.

train.describe()
# Tipos dos dados.

train.info()
# Qual a quantidade de sobreviventes do Titanic?

survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]



print(f'Total number of passengers: {len(train)}')

print(f'Survived: {len(survived)}')

print(f'Not Survived: {len(not_survived)}')
# Os passageiros do Titanic estavam divididos em 3 classes:

# os que viajavam na 1ª classe na sua maioria eram os mais ricos, 

# na 2ª classe eram viajantes de classe média,

# os da 3ª classe eram principalmente imigrantes. 



# Total de passageiros por classe.

train.groupby('Pclass').size()
# Número de passageiros de cada classe e seus respectivos sexos.

pd.crosstab(train['Pclass'],train['Sex'])
# Número de sobreviventes por classe.

# 0 - not survived    1 - survived

pd.crosstab(train['Pclass'],train['Survived'])
# Média de sobreviventes por classe.

train[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
# Quantidade de passageiros de acordo com o sexo.

# female = feminino, male = masculino

train['Sex'].value_counts()
# Número de sobreviventes por sexo.

# 0 - not survived    1 - survived

pd.crosstab(train['Sex'],train['Survived'])
# Média de sobreviventes de cada sexo.

train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.countplot(x='Survived' ,hue='Sex', data=train, palette='rainbow')
# Número de passageiros de acordo com a país de embarque.

# S-> Southampton (Reino Unido)

# C-> Cherbourg-Octeville (França)

# Q-> Queenstown (Irlanda)

train['Embarked'].value_counts()
# Número de sobreviventes por país de embarque.

# 0 - not survived    1 - survived

pd.crosstab(train['Embarked'],train['Survived'])
# Média de sobreviventes de acordo com o país que ele embarcou.

train[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.barplot(x='Embarked', y='Survived', data=train, palette='rainbow')
# Número de pais e filhos viajando com o passageiro.

train['Parch'].value_counts()
# Número de sobreviventes de acordo com a quantidade de pais e filhos viajando com o passageiro.

# 0 - not survived    1 - survived

pd.crosstab(train['Parch'],train['Survived'])
# Média de sobrevivência com base no número de pais/ filhos que o passageiro tinha no navio.

train[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.barplot(x='Parch', y='Survived', data=train, palette='rainbow')
# Número de irmãos e cônjuges viajando com o passageiro.

train['SibSp'].value_counts()
# Número de sobreviventes de acordo com a quantidade de irmãos e cônjuges viajando com o passageiro.

# 0 - not survived    1 - survived

pd.crosstab(train['SibSp'],train['Survived'])
# Média de sobrevivência com base no número de irmãos e cônjuges que o passageiro tinha no navio.

train[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.barplot(x='SibSp', y='Survived', data=train, palette='rainbow')
# Distribuição das idades.

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

fig = train.Age.hist(bins=25)

fig.set_title('Distribuição da idade')
# Distribuição dos valores da passagem.

plt.figure(figsize=(12,6))

plt.subplot(1,2,2)

fig = train.Fare.hist(bins=25)

fig.set_title('Distribuição do valor da passagem')
# Matriz de correlação.

# Possibilita a análise simultânea da associação entre variáveis.

plt.figure(figsize=(15,6))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
# Verificando a quantidade de valores nulos no dataset de treino.

train.isnull().sum().sort_values(ascending=False)
# Verificando a quantidade de valores nulos no dataset de teste.

test.isnull().sum().sort_values(ascending=False)
# Preenchendo os valores nulos da coluna 'Age'.

for dataset in data:

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
# Preenchendo os valores nulos da coluna 'Fare'.

for dataset in data:

    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)
# A coluna possui um número bem pequeno de valores faltantes,

# então podemos preencher-la com o valor mais frequente, 

# Variável que mais se repete na coluna 'Embarked'

train['Embarked'].describe()
# Preenchendo os valores nulos da coluna 'Embarked'.

top = 'S'

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(top)



train.isnull().sum().sort_values(ascending=False)
# Convertendo a coluna 'Embarked' para númerico.

ports = {"S": 0, "C": 1, "Q": 2}



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
# Convertendo a coluna 'Sex' para númerico.

genders = {"male": 0, "female": 1}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
# Podemos observar que os nomes estão acompanhados de títulos, vamos extrair esses títulos para saber se eles tem alguma correlação entre as variáveis.

for dataset in data:

    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

train.head()
# Iremos agrupar esses títulos nas seguintes categorias:



# Miss -> Mulheres jovens.

# Mr -> Homens casados.

# Mrs -> Mulheres casadas.

# Master -> Homens jovens.

# Rare -> Capitão,médicos,reverendos e pessoas da alta realeza.



# Definindo valores para cada título.

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Lady','the Countess','Don','Sir','Jonkheer','Capt','Col','Dr','Major','Rev','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].map(titles)

    dataset['Title'] = dataset ['Title']. fillna (0) 



# Taxa de sobrevivência com base nos títulos.

train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
sns.barplot(x='Title', y='Survived', data=train, palette='rainbow')
# Vamos unir a coluna 'SibSp' e 'Parch' para criar uma nova coluna 'Family'.

# Tudo indica que essas duas colunas juntas nos dão o número de membros de uma mesma família.

for dataset in data:

    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

    

# Média de sobrevivência com base no número de membros da família no navio. 

train[['Family','Survived']].groupby(['Family'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.barplot(x='Family', y='Survived', data=train, palette='rainbow')
# Vamos verificar se quem viajava sozinho tinha mais chance de sobreviver.

for dataset in data:

    dataset['Alone'] = dataset['Family'].apply(lambda x: x if x == 1 else 0)



# Média de sobrevivência com base no número de membros da família no navio.

train[['Alone','Survived']].groupby(['Alone'], as_index = False).mean().sort_values(by='Survived',ascending=False)
sns.barplot(x='Alone', y='Survived', data=train, palette='rainbow')
# Verificando os Outliers

# Os valores fora da barreira externa são caracterizados como Outliers.

sns.boxplot(data = train.drop('PassengerId',axis=1), orient= 'h')
# Quartil inferior e superior coluna 'Age'

def quartil():

    Q1 = train['Age'].quantile(q= 0.25)

    Q3 = train['Age'].quantile(q= 0.75)

    print(f'Q1 = {Q1} e Q3 = {Q3}')

    # Calculando a barreira externa.

    amp = Q3 - Q1

    limite_max = Q3 + 3 * amp

    limite_min = Q1 - 3 * amp

    print(f'Os Outliers da coluna Age estão entre {limite_min} e {limite_max}')

quartil()
# Substituindo os valores.

for dataset in data:

    dataset['Age'] = np.where(dataset['Age'] > 73, 73, dataset['Age'])
# Quartil inferior e superior coluna 'Fare'

def quartil():

    Q1 = train['Fare'].quantile(q= 0.25)

    Q3 = train['Fare'].quantile(q= 0.75)

    print(f'Q1 = {Q1} e Q3 = {Q3}')

    # Calculando a barreira externa.

    amp = Q3 - Q1

    limite_max = Q3 + 3 * amp

    limite_min = Q1 - 3 * amp

    print(f'Os Outliers da coluna Fare estão entre {limite_max} e {limite_min}')

quartil()
# Substituindo os valores.

for dataset in data:

    dataset['Fare'] = np.where(dataset['Fare'] > 99, 99, dataset['Fare'])
# Visualização após o tratamento dos Outliers.

# Deixarei alguns Outliers para obtermos um modelo mais generalista.

sns.boxplot(data = train.drop('PassengerId',axis=1), orient= 'h')
# Separando os dados de treino e teste

X_train, X_test, y_train,y_test = train_test_split(train[['Pclass','Alone','Fare','Title','Embarked','Age','SibSp','Parch','Family','Sex']], train['Survived'],

test_size=0.3, random_state = 7)

X_train.shape,X_test.shape
# Extração de variáveis com testes estatísticos univariados (Teste qui-quadrado)

f_score = chi2(X_train, y_train)

f_score
# Listando o P-values por variável

p_values = pd.Series(f_score[1])

p_values.index = X_train.columns

p_values.sort_values(ascending = False)
# Selecione como preditoras as variáveis com os menores valores de p_values, pois para o p-value, quanto menor o valor, melhor.

# Esse critério se encaixa bem para datasets pequenos

predictors = ['Pclass','Alone','Fare','Title','SibSp']
X_train = train[predictors].values

X_test = test[predictors].values

y_train = train['Survived'].values
# Abordagem probabilística (Teorema de Bayes)

gaussian = GaussianNB()

gaussian.fit(X_train,y_train)

prev_gaussian = gaussian.predict(X_test)

result_train = gaussian.score(X_train, y_train)

result_val = cross_val_score(gaussian,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

prev_logreg = logreg.predict(X_test)

result_train = logreg.score(X_train, y_train)

result_val = cross_val_score(logreg,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=7)

decision_tree.fit(X_train,y_train)

prev_tree = decision_tree.predict(X_test)

result_train = decision_tree.score(X_train, y_train)

result_val = cross_val_score(decision_tree,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)

knn.fit(X_train,y_train)

prev_knn = knn .predict(X_test)

result_train = knn.score(X_train, y_train)

result_val = cross_val_score(knn,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
random_forest = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=7)

random_forest.fit(X_train,y_train)

prev_random = random_forest.predict(X_test)

result_train = random_forest.score(X_train, y_train)

result_val = cross_val_score(random_forest,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
xboost = XGBClassifier()

xboost.fit(X_train, y_train)

prev_xboost = xboost.predict(X_test)

result_train = xboost.score(X_train, y_train)

result_val = cross_val_score(xboost,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
svc = SVC(kernel = 'rbf', random_state = 7, C = 10.0, gamma='auto')

svc.fit(X_train,y_train)

previsoes = svc.predict(X_test)

result_train = svc.score(X_train, y_train)

result_val = cross_val_score(svc,X_train, y_train, cv=5).mean()

print(f'taring score = {result_train}, while validation score = {result_val}')
submission = pd.DataFrame({

        'PassengerId': test['PassengerId'],

        'Survived': prev_xboost})
submission.to_csv('submission.csv', index=False)