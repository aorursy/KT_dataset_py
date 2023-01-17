# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
wineDataBase = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wineDataBase.shape
wineDataBase.head()
wineDataBase.info()
#Criando o target categorico e outra variavel categorica

wineDataBase['qualidade_binaria'] = ['bom' if x > 6.5 else 'ruim' for x in wineDataBase['quality']]
#Contando os valores do target

wineDataBase['qualidade_binaria'].value_counts()
#Mudando o nome das colunas para facilitar a visualização

wineDataBase.columns = ['fix_ac', 'vol_ac', 'cit_ac', 'res_sugar',

              'clohrid', 'fre_diox', 'tot_diox', 'density',

              'ph', 'sulph', 'alcool', 'qualidade','qualidade_binaria' ]
wineDataBase['media_ph'] = ['maior' if x > 3.31113 else 'menor' for x in wineDataBase['ph']]

#Fazendo encoding com a string

wineDataBase['target_encode'] = [1 if x == 'bom' else 0 for x in wineDataBase['qualidade_binaria']]
#Criando o X e y

X = wineDataBase.drop(columns = ['qualidade_binaria', 'media_ph', 'target_encode', 'qualidade'])

y = wineDataBase['target_encode']

y_reg = wineDataBase['qualidade']
#Countplot

plt.figure(figsize=(12,6))

sns.countplot('qualidade_binaria', data = wineDataBase)
#Pairplot com a discriminante

sns.pairplot(wineDataBase, hue='qualidade_binaria')
#heatmap

plt.figure(figsize=(12,6))

sns.heatmap(wineDataBase.corr(), annot = True)

plt.show()
#Visualização da importancia das features usando Shapiro

from yellowbrick.features import Rank1D

# Criando o visualizador

visualizer = Rank1D(algorithm='shapiro')

plt.figure(figsize=(10,10))

visualizer.fit(X, y)           # Fit nos dados para o visualizer

visualizer.transform(X)        # Tranformando os dados

visualizer.show()              # Imprimindo as figuras
from yellowbrick.target import FeatureCorrelation



#Visualizador de correlacao com o target

features = list(X.columns)

visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X, y)        

visualizer.show()
#Lineplot

plt.figure(figsize=(12,6))

sns.lineplot(x = wineDataBase.index, hue = 'qualidade_binaria', y ='alcool', data = wineDataBase)

plt.show()
#Lineplot com duas distribuições

plt.figure(figsize=(12,6))

sns.lineplot(x = wineDataBase.index, y ='sulph', data = wineDataBase)

sns.lineplot(x = wineDataBase.index, y ='alcool', data = wineDataBase)

plt.show()
# Grafico de regressao com 1 variavel

sns.lmplot(x= 'alcool', y = 'sulph',  data = wineDataBase, height=8)
from yellowbrick.features import Rank2D



# Criando o visualizador por correlacao de pearson

visualizer = Rank2D(algorithm='pearson')

visualizer.fit(X, y)           # Fazendo o ajuste nos dados

visualizer.transform(X)        # Transformando os dados

visualizer.show()              # Mostrando o gráfico
#Usando o visualizador por covariancia

visualizer = Rank2D(algorithm='covariance')



visualizer.fit(X, y)           

visualizer.transform(X)        

visualizer.show()
#Gráfico das principais componenentes

from yellowbrick.features.pca import PCADecomposition



plt.figure(figsize= (15,7))

visualizer = PCADecomposition(scale=True, proj_features=True)

visualizer.fit_transform(X, y)

visualizer.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#Importando os pacotes

from sklearn.preprocessing import LabelEncoder

#Trazendo a variavel media_ph de volta para o dataframe

X = wineDataBase.drop(columns = ['qualidade_binaria', 'target_encode', 'qualidade'])

y = wineDataBase['target_encode']

y_reg = wineDataBase['qualidade']



#Aplicando nova divisão em base de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

#Importando os pacotes

from sklearn.preprocessing import LabelEncoder

#Instanciando o objeto

lab = LabelEncoder()

#Ajustando e transformando os dados

X_train['media_ph'] = lab.fit_transform(X_train['media_ph'])

X_test['media_ph'] = lab.fit_transform(X_test['media_ph'])
X_train['media_ph'].head() 
#Importando os pacotes

from tpot import TPOTRegressor
#Instanciando o objeto

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
#Ajustando os dados

# Fazendo o train test split com o y para regressao

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, random_state = 42, test_size = 0.33)

#Aplicando label encoding na média pH

X_train['media_ph'] = lab.fit_transform(X_train['media_ph'])

X_test['media_ph'] = lab.fit_transform(X_test['media_ph'])

#Ajustando aos dados

tpot.fit(X_train,y_train)
tpot.export('../output/pipeline_tpot_regressao.py')
