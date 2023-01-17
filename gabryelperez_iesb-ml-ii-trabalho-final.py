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
# Importando as classes

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Carregando os dados

df = pd.read_csv('../input/zoo.csv')

ani_class = pd.read_csv('../input/class.csv')
# Visualização prévia dos dados

df.head()
# Verificando os tipos e os valores nulos

df.info()
# Verificando a quantidade ṕr classe

sns.countplot(df['class_type'])

pd.Series.value_counts(df['class_type'])

# 1 	41	Mammal

# 2 	20	Bird

# 3 	5	Reptile

# 4 	13	Fish

# 5 	4	Amphibian

# 6 	8	Bug

# 7 	10	Invertebrate
# Join tabela de animais e tabela de classes para exibir nome das classes

dfu = pd.merge(df,ani_class,how='left',left_on='class_type',right_on='Class_Number')

dfu.head()
# Verificando a que classe a maioria dos animais de zoológico pertence

sns.factorplot('Class_Type', data=dfu,kind="count", aspect=2)
# Verificando a estatística

df.describe()
# Criando mapa de calor para mostrar as correlações

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("Correlation Heatmap")

corr = df.corr()

sns.heatmap(corr, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Mostrando as correlações que são superior a 0,7 (positiva ou negativa)

corr[corr != 1][abs(corr)> 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)
# Dividindo o DataFrame

from sklearn.model_selection import train_test_split



# Treino e teste

train, test = train_test_split(df, test_size=0.30, random_state=42)



# Verificando o tanho dos DataFrames

train.shape, test.shape
# Selecionado as features

removed_cols = ['animal_name', 'class_type']

feats = [c for c in train.columns if c not in removed_cols]
# Importando as classes

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.metrics import accuracy_score
# Selecionando os modelos a serem testados

models = {'RandomForest': RandomForestRegressor(random_state=42),

          'ExtraTrees': ExtraTreesRegressor(random_state=42),

          'GradientBoosting': GradientBoostingRegressor(random_state=42),

          'DecisionTree': DecisionTreeRegressor(random_state=42),

          'AdaBoost': AdaBoostRegressor(random_state=42),

          'KNN 1': KNeighborsRegressor(n_neighbors=1),

          'KNN 3': KNeighborsRegressor(n_neighbors=3),

          'KNN 11': KNeighborsRegressor(n_neighbors=11),

          'SVR': SVR(),

          'Linear Regression': LinearRegression()}
# Criando a função para execução dos modelos

def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(test[feats])

    return accuracy_score(test[y_name], preds.round())
# Executando os modelos

scores = []

for name, model in models.items():

    score = run_model(model, train, test, feats, 'class_type')

    scores.append(score)

    print(name, ':', score) 
# Apresentando as acurácias dos modelos no gráfico de barras

pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()
# Rodando o modelo

knn1 = KNeighborsRegressor(n_neighbors=1)

knn1.fit(train[feats], train['class_type'])

preds = knn1.predict(test[feats])

accuracy_score(test['class_type'], preds.round())
# Submetendo os resultados

test['class_type'] = knn1.predict(test[feats]).astype(int)

test[['animal_name', 'class_type']].to_csv('submission.csv', index=False)