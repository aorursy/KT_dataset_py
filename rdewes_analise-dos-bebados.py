# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/student-por.csv')

df.head()
df.info()
df['sex'].value_counts(normalize=True)
sns.catplot(x='school', hue='sex', data=df, kind='count', height=6, aspect=2)
# Correlação entre variáveis numéricas

plt.figure(figsize=(15, 15))

hmap = sns.heatmap(df.corr(), square=True, annot=True, linewidth=0.5)
# Verificando a relação entre a disponibilidade de internet e a nota G3,

# separada por área urbana e rural.



sns.swarmplot(x='internet', y='G3', hue='address', data=df)
# Verificando a relação entre a disponibilidade de Alcool Semanal e a nota G3,

# separada por genero.



sns.swarmplot(x='Dalc', y='G3', hue='sex', data=df)
# Verificando a relação entre a disponibilidade de Alcool Fim de Semana e a nota G3,

# separada por genero.



sns.swarmplot(x='Walc', y='G3', hue='sex', data=df)
# Distribuição da Nota Final

sns.distplot(df['G3'], kde=True)
# Modificando o tipo escola



df.loc[df['school'] == 'GP', 'school'] = 1

df.loc[df['school'] == 'MS', 'school'] = 0
df['school'].value_counts()
# Modificando o tipo escola



df.loc[df['sex'] == 'M', 'sex'] = 1

df.loc[df['sex'] == 'F', 'sex'] = 0



# Modificando o endereço



df.loc[df['address'] == 'U', 'address'] = 1

df.loc[df['address'] == 'R', 'address'] = 0



# Modificando o famsize



df.loc[df['famsize'] == 'GT3', 'famsize'] = 1

df.loc[df['famsize'] == 'LE3', 'famsize'] = 0



# Modificando o Pstatus



df.loc[df['Pstatus'] == 'T', 'Pstatus'] = 1

df.loc[df['Pstatus'] == 'A', 'Pstatus'] = 0



# Transformando as variáveis yes/no



for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',

           'romantic']:

    df[col] = df[col].map({'yes': 1, 'no': 0})



# Transformando as variáveis categóricas

for col in ['Mjob', 'Fjob', 'reason', 'guardian']:

    df[col] = df[col].astype('category')

    df[col] = df[col].cat.codes
df.info()
df.head()
# Criar conjuntos de treino e teste



# Importando a função do sklearn

from sklearn.model_selection import train_test_split



# Separar os dados com 20% para teste e usando uma semente aleatória

# para podermos replicar o resultado

train, test = train_test_split(df, test_size=0.2, random_state=42)
# Verificando o tamanho

train.shape, test.shape
# Listar colunas que não serão usadas

remove = ['G1', 'G2', 'G3']



# Lista das caracteristicas para o modelo

feats = [col for col in train.columns if col not in remove]

# Criando o modelo de regressão linear

from sklearn import linear_model

regr = linear_model.LinearRegression()
# treinar o modelo

regr.fit(X=train[feats], y=train['G3'])
# Gerar as predições

preds = regr.predict(X=test[feats])
plt.scatter(test['G3'], preds, color='blue')

#plt.plot(preds, test['G3'], color='blue', linewidth=3)

#plt.xticks(())

#plt.yticks(())



plt.show()
from sklearn.metrics import r2_score

r2_score(test['G3'], preds)
from sklearn.metrics import mean_squared_error

mean_squared_error(test['G3'], preds)