# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.describe()
print('Quantidade de dados Nulos em cada variável')

df.isna().sum()
plt.figure(figsize=(10,4))

sns.countplot(df['target'], palette='Pastel1')

plt.title('Distribuição entre Doentes e não-doentes')
plt.figure(figsize=(10,4))

sns.countplot(df['sex'], palette='Set2')

plt.title('Distribuição por Gênero')
plt.figure(figsize=(12,6))

(df['age']).hist(bins=20)

plt.title('Distribuição de Frequência por Idade')

plt.xlabel('Idade')

plt.ylabel('Freq')
plt.figure(figsize=(12,6))

sns.countplot(x='age', data=df, hue='sex',palette='Pastel1')

plt.title('Quantidade de Pacientes por idade e Sexo')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True, cmap='OrRd')

plt.title('Matriz de Correlação entre as variáveis')
sns.pairplot(df)
#importando a biblioteca para separar em treino e teste 

from sklearn.model_selection import train_test_split
#separando o dataset

X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Regressão Logística

from sklearn.linear_model import LogisticRegression
#instanciando o modelo

logmodel = LogisticRegression()
#treinando o modelo

logmodel.fit(X_train, y_train)
#Fazendo as predições

pred = logmodel.predict(X_test)
# Importando métricas de avaliação do modelo

from sklearn.metrics import classification_report
#Avaliando modelo

print(classification_report(y_test, pred))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier
#Instanciando o modelo

dtree = DecisionTreeClassifier()
#Treinando o modelo

dtree.fit(X_train, y_train)
#Fazendo as predições

dpred = dtree.predict(X_test)
#Avaliando o modelo

from sklearn.metrics import confusion_matrix

print(classification_report(y_test, dpred))

print('\n')

dtree_matrix = (confusion_matrix(y_test, dpred))
#Matriz de Confusão

from sklearn.metrics import confusion_matrix

dtree_matrix = pd.DataFrame(dtree_matrix)

class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(dtree_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Matriz de Confusão Dtree', y = 1.1)

plt.ylabel('Valor Atual')

plt.xlabel('Valor Predito')

plt.show()
#Random Forest

from sklearn.ensemble import RandomForestClassifier
#Instanciando o modelo

rfc = RandomForestClassifier(n_estimators=200, 

min_samples_split=5, 

max_depth=12,

random_state=42,

n_jobs=-1)
#Treinando o modelo

rfc.fit(X_train, y_train)
#Fazendo as predições

rfpred = rfc.predict(X_test)
# Avaliando o modelo

#Avaliando o modelo

from sklearn.metrics import confusion_matrix

print(classification_report(y_test, rfpred))

print('\n')

rfc_matrix = (confusion_matrix(y_test, rfpred))
#Matriz de Confusão

from sklearn.metrics import confusion_matrix

rfc_matrix = pd.DataFrame(rfc_matrix)

class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



sns.heatmap(pd.DataFrame(rfc_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Matriz de Confusão Dtree', y = 1.1)

plt.ylabel('Valor Atual')

plt.xlabel('Valor Predito')

plt.show()
#Acurácia dos modelos

from sklearn.metrics import accuracy_score

print('Regressão Logistica')

accuracy_score(y_test,pred)

print('Decision Tree')

accuracy_score(y_test,dpred)
print('Random Forest')

accuracy_score(y_test,rfpred)