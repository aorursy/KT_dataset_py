import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt   
import seaborn as sns  
import os as os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# permitir visualizar todas as colunas
pd.options.display.max_columns = None
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))
display(train.dtypes)
display(train.head())
train.describe()
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


df_merged.isnull().sum()
# age
age_median = df_merged['Age'].median()
df_merged['Age'].fillna(age_median, inplace=True)

# fare
fare_median = df_merged['Fare'].median()
df_merged['Fare'].fillna(fare_median, inplace=True)

# embarked
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)
# converter 'Sex' em 0 e 1
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})

# dummie variables para 'Embaked'
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)

display(df_merged.head())
# recuperar datasets de treino e teste
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]
from sklearn.linear_model import LogisticRegression
# criar um modelo de Regressão Logística
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train, target)

# verificar a acurácia do modelo
acc_logReg = round(lr_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(acc_logReg))
from sklearn.tree import DecisionTreeClassifier
# criar um modelo de árvore de decisão
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(train, target)

# verificar a acurácia do modelo
acc_tree = round(tree_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acc_tree))
# declarar os valores das variáveis para mim e minha esposa
fagner_paes = np.array([2, 0, 36, 1, 1, 32.2, 0, 0, 0, 1]).reshape((1, -1))
esposa = np.array([1, 1, 40, 1, 1, 32.2, 0, 0, 0, 1]).reshape((1,-1))

# verificar se nós teríamos sobrevivido
print("Fagner Paes:\t{}".format(tree_model.predict(fagner_paes)[0]))
print("Su:\t{}".format(tree_model.predict(esposa)[0]))