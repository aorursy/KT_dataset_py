#nesta estapa impostamos as bibliotecas necessarias para execução do codigo.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt   
import seaborn as sns  
import os as os
#foi feito uma estrutura de laços para capitar todos os arquivos referente ao projeto Titanic importados da base de dados do Kaggle.

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# permitir visualizar todas as colunas
pd.options.display.max_columns = None
#Aqui atribuo os arquivos train e test atraves do comando read da biblioteca pandas.
#O "train" seria um arquivo de treino para otimizar o algoritmo de predição.
#O "test" seria o arquivo para validarmos o nosso treino e conferir a assertividade.

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#mostra a quantidade de variavel e de entradas utilizadas
print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))
#este comando exibe os tipos de variaveis.
display(train.dtypes)
#Este comando "head" exibe as primeiras linhas da base de dados. (Isto é feito para olharmos os dados e termos uma noção do que fazermos com ele).

display(train.head())
#Este comando exibe calculos basicos envolvendo os valore snumeros do banco de dados.

train.describe()
#Isso são graficos Histograma, ele exibe de forma visual a concentração de dados conforme os valores das variaveis.

train.hist(figsize=(10,8));
# salvar os índices dos datasets para recuperação posterior
train_idx = train.shape[0]
test_idx = test.shape[0]

# salvar PassengerId para submissao ao Kaggle
passengerId = test['PassengerId']

# extrair coluna 'Survived' e excluir ela do dataset treino
target = train.Survived.copy()
train.drop(['Survived'], axis=1, inplace=True)

# concatenar treino e teste em um único DataFrame
df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))
#Exibir as 5 primeiras entradas do conjunto de dados antes de remover features.
display(df_merged.head())

#Função drop remove as colunas do DataFrame.
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Exibir as 5 primeiras entradas do conjunto de dados depois de remover features.
display(df_merged.head())

#Basicamente faço a junção das informações e removo as colunas a qual não agregam positivamente.
df_merged.isnull().sum()
# age (idade)
age_median = df_merged['Age'].median()
df_merged['Age'].fillna(age_median, inplace=True)

# fare (valor passagem)
fare_median = df_merged['Fare'].median()
df_merged['Fare'].fillna(fare_median, inplace=True)

# embarked (embarcou)
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)
# converter 'Sex' em 0 e 1 (A qual masculuno é representado pelo 0 e feminino representado pelo valor 1)
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})

# dummie variables para 'Embaked'
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)

display(df_merged.head())

#As primeiras linhas depois de remover as colunas e adicionado o ponto de embarque.
# recuperar datasets de treino e teste
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]
from sklearn.tree import DecisionTreeClassifier
# criar um modelo de árvore de decisão
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(train, target)

# verificar a acurácia do modelo
acc_tree = round(tree_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acc_tree))

#Mede a precisão do modelo de árvore de decisão.
from sklearn.linear_model import LogisticRegression
# criar um modelo de Regressão Logística
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train, target)

# verificar a acurácia do modelo
acc_logReg = round(lr_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(acc_logReg))

#Mede a precisão do modelo Regressão Logística.
#Teste sobrevivencia.
sidney = np.array([2, 0, 30, 1, 1, 891, 0, 0, 0, 1]).reshape((1, -1))
andressa = np.array([1, 1, 25, 1, 1, 891, 0, 0, 0, 1]).reshape((1,-1))

# verificar se nós teríamos sobrevivido
print("Sidney:\t{}".format(tree_model.predict(fagner_paes)[0]))
print("Andressa:\t{}".format(tree_model.predict(esposa)[0]))