# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importando bibliotecas que serao utilizadas neste projeto

import seaborn as sns

import itertools

import imblearn

import matplotlib.pyplot as plt

%matplotlib inline



# Misc

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

import pandasql as ps



# Models

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.set_option('display.max_columns', None)

import pickle

import gc
train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_treino.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'measurement_number': np.int16

                               ,'orientation_X': np.float32

                               ,'orientation_X': np.float32

                                ,'orientation_Y': np.float32

                                ,'orientation_Z': np.float32

                                ,'orientation_W': np.float32

                                ,'angular_velocity_X': np.float32

                                ,'angular_velocity_Y': np.float32

                                ,'angular_velocity_Z': np.float32

                                ,'linear_acceleration_X': np.float32

                                ,'linear_acceleration_Y': np.float32

                                ,'linear_acceleration_Z': np.float32

                            })



test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_teste.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'measurement_number': np.int16

                               ,'orientation_X': np.float32

                               ,'orientation_X': np.float32

                                ,'orientation_Y': np.float32

                                ,'orientation_Z': np.float32

                                ,'orientation_W': np.float32

                                ,'angular_velocity_X': np.float32

                                ,'angular_velocity_Y': np.float32

                                ,'angular_velocity_Z': np.float32

                                ,'linear_acceleration_X': np.float32

                                ,'linear_acceleration_Y': np.float32

                                ,'linear_acceleration_Z': np.float32

                            })



y_train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/y_treino.csv'

                            ,dtype = {

                                'series_id': np.int16

                               ,'group_id': np.int16

                            })



train.shape, test.shape, y_train.shape
# Realizando o merge.

# Já eliminando as linhas que tem no dataset 'y_train' mas nao tem no dataset 'train'

df = pd.merge(train, y_train, on='series_id', how='left')

df.shape
# Cria um label encoder object

le = preprocessing.LabelEncoder()

suf="_le"



# Iteracao para cada coluna do dataset de treino

for col in df:

    if df[col].dtype == 'object':

        le.fit_transform(df[col].astype(str))

        df[col+suf] = le.transform(df[col])
# Split features and labels

X = df.drop(['surface', 'surface_le', 'row_id_le', 'row_id', 'group_id'],axis=1)

y = df['surface_le']



# Aplicando a mesma escala nos dados

X = MinMaxScaler().fit_transform(X)



# Padronizando os dados (0 para a média, 1 para o desvio padrão)

X = StandardScaler().fit_transform(X)



# Verificando o shape apos o split entre feature e target

X.shape, y.shape

df.head(5)
# Definindo os valores para o número de folds

num_folds = 12

seed = 13



# Separando os dados em folds

kfold = KFold(num_folds, True, random_state = seed)



# Criando o modelo

modeloCART = DecisionTreeClassifier()



# Cross Validation

resultado = cross_val_score(modeloCART, X, y, cv = kfold, scoring = 'accuracy')



# Print do resultado

print("Acurácia: %.3f" % (resultado.mean() * 100))



# Treinando o modelo

modeloCART.fit(X, y)
# Configurando o dataset de teste, retirando algumas colunas 

X_final = test.drop(['row_id'],axis=1)

X_final = MinMaxScaler().fit_transform(X_final)

X_final = StandardScaler().fit_transform(X_final)



# Fazendo as previsoes de surface no dataset de teste

predCART = modeloCART.predict(X_final)



# Voltando a transformacao da variavel target em formato texto

surface_pred = le.inverse_transform(predCART)
#Gerando Arquivo de Submissao

submission = pd.DataFrame({

    "series_id": test.series_id, 

    "surface": surface_pred

})



# Removendo registros duplicados

submission = submission.drop_duplicates()
# Executando query para identificar as superficies com maiores quantidade, para fazer o submit

q1 = """ SELECT x.series_id, x.surface, MAX(x.qtde) maior

           FROM (SELECT series_id, surface, count() as qtde

                   FROM submission

                  GROUP BY series_id, surface) x

          GROUP BY x.series_id"""



sub_final = ps.sqldf(q1, locals())

sub_final = sub_final.drop(['maior'],axis=1)
# Salvando o resultado das previsoes em um arquivo .csv

sub_final.to_csv('submission.csv', index=False)