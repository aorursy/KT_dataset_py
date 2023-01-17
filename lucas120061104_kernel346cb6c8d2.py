import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

import re
print(os.listdir("../input"))
treino = pd.read_csv('../input/train.csv',index_col=0)

val = pd.read_csv('../input/test.csv',index_col=0)
treino['nota_mat'] = treino['nota_mat']

treino.head()

val.head()
treino['base'] = 'TREINO'

val['base'] = 'VALIDAÇÃO'

cod_mun_val = val['codigo_mun']
geral = pd.concat([val,treino])

geral.shape
geral.head()
variaveis = [c for c in geral.columns if c not in ['codigo_mun','comissionados_por_servidor','ranking_igm']] 

variaveis

geral = geral[variaveis]

geral.dtypes
lista_caract = ['area','densidade_dem']



for i in lista_caract:

    geral[i] = geral[i].str.replace(',','').astype('float')    

cat_dict = {}

for c in geral.select_dtypes('object').columns:

    cat_dict[c] = geral[c].astype('category').cat.categories
cat_dict

    
geral.head()
variaveis = geral.columns[geral.dtypes == 'object']

for i in variaveis:

        geral[i] = geral[i].astype('category').cat.codes
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
geral.head()
#geral.fillna(-2, inplace=True)

geral = geral.fillna(geral.mean())

val = geral[geral['base'] == 1]

treino = geral[geral['base'] == 0]

val.shape,treino.shape
train, valid = train_test_split(treino,random_state=42)

nomes=[]

for i in train.columns:

    if i != 'nota_mat':

        nomes.append(i)





erros = []

for j in list(range(1,1000)):

    rf = RandomForestRegressor(random_state = j)

    rf.fit(train[nomes],train['nota_mat'])

    erros.append(mean_squared_error(rf.predict(valid[nomes]),valid['nota_mat'])**(1/2))

menor =erros.index(min(erros))

rf = RandomForestRegressor(random_state = menor+1)

rf.fit(train[nomes],train['nota_mat'])

mean_squared_error(rf.predict(valid[nomes]),valid['nota_mat'])**(1/2)

predictions = rf.predict(val[nomes])



cod_mun_val = cod_mun_val.str.extract('([0-9]{1,}$)')

cod_mun_val
arquivo = pd.DataFrame({'codigo_mun':cod_mun_val[0],'nota_mat':predictions})
arquivo
nome_arq = 'nota_mat.csv'



arquivo.to_csv(nome_arq,index=False)



print('Arquivo salvo: ' + nome_arq)