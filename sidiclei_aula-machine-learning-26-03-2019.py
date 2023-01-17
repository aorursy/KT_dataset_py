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
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/student-por.csv')

df.head()
# Verificando o tamanho linhas e colunas

df.shape
# tamanho, tipo, dados faltantes, etc

df.info()
# quantidade de homens e mulheres

df['sex'].value_counts(normalize=True)
# Gráfico

sns.catplot(x='school', hue='sex', data=df, kind='count', height=5,aspect=3)
plt.figure(figsize=(15,15))

hmap = sns.heatmap(df.corr(), square=True, annot=True, linewidths=0.5)
# Verificando a relação entre a disponibilidade de internet e a nota G3

# separado por área urbana e rural

plt.figure(figsize=(15,10))

sns.swarmplot(x='internet', y='G3', hue='address', data=df)
# relação do consumo de alcool durante a semana e a nota final por sexo

plt.figure(figsize=(15,10))

sns.swarmplot(x='Dalc', y='G3', hue='sex', data=df)
# relação do consumo de alcool no final de semana e a nota final por sexo

plt.figure(figsize=(15,10))

sns.swarmplot(x='Walc', y='G3', hue='sex', data=df)
plt.figure(figsize=(15,10))

sns.distplot(df['G3'], kde=True)
df.head().T
# Transformar as variáveis em binárias



df[df['school'] == 'GP']['school'] = 1

df[df['school'] == 'LS']['school'] = 0
df.info()
# Transformando colunas string em numéricas



df.loc[df['school'] == 'GP','school'] = 1

df.loc[df['school'] == 'MS','school'] = 0



df.loc[df['sex'] == 'M','sex'] = 1

df.loc[df['sex'] == 'F','sex'] = 0



df.loc[df['address'] == 'U','address'] = 1

df.loc[df['address'] == 'R','address'] = 0



df.loc[df['famsize'] == 'GT3','famsize'] = 1

df.loc[df['famsize'] == 'LE3','famsize'] = 0



df.loc[df['Pstatus'] == 'T','Pstatus'] = 1

df.loc[df['Pstatus'] == 'A','Pstatus'] = 0

# Transformando as variáveis yes/no



for col in ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']:

    df[col] = df[col].map({'yes':1,'no':0})
# Transformar diversas categorias

for col in ['Mjob','Fjob','reason','guardian']:

    df[col] = df[col].astype('category')

    df[col] = df[col].cat.codes
df.info()
df.head()
# Vamos criar conjunto de treino e teste



# Importando a função do sklearn

from sklearn.model_selection import train_test_split

# Separar os dados: 20% teste - usando semente aleatória

# para podermos replicar o resultado

train, test = train_test_split(df, test_size=0.2,random_state=42)
# verificando o tamanho

train.shape, test.shape
# Vamos selecionar as colunas a serem usadas



# Lista com as colunas que não serão usadas



remove = ['G1','G2','G3']



# Lista das caracteristicas para o modelo

feats = [col for col in train.columns if col not in remove]
# Criando o modelo de regressão linear

from sklearn import linear_model



regr = linear_model.LinearRegression()
# Treinar o modelo

regr.fit(X=train[feats],y=train['G3'])
# Gerar as predições

preds = regr.predict(X=test[feats])
# Avaliando o resultado con Mean Square Error

from sklearn.metrics import mean_squared_error



# Chamamos a função passando y predito e o y real

mean_squared_error(test['G3'], preds)
# Avaliando com r2 score

from sklearn.metrics import r2_score



r2_score(test['G3'],preds)
remove = ['G3']



# Lista das caracteristicas para o modelo

feats = [col for col in train.columns if col not in remove]



from sklearn import linear_model



regr = linear_model.LinearRegression()



# Treinar o modelo

regr.fit(X=train[feats],y=train['G3'])



# Gerar as predições

preds = regr.predict(X=test[feats])



# Avaliando o resultado con Mean Square Error

from sklearn.metrics import mean_squared_error



# Chamamos a função passando y predito e o y real

print('mean_squared_error: ', mean_squared_error(test['G3'], preds))



# Avaliando com r2 score

from sklearn.metrics import r2_score



print('r2_score: ',r2_score(test['G3'],preds))