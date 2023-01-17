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
#Carregar os datasets

df = pd.read_csv('../input/train.csv')

ts = pd.read_csv('../input/test.csv')
#Analise estatística da base

df.describe()
#Realiza a união das bases

df = df.append(ts)
#Tratamento o campo codigo_mun, retirada do ID_ID_

df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_', '')
#Criação da lista dos campos que serão considerados na predição

lista = ['estado','municipio','porte','populacao','pib','anos_estudo_empreendedor','jornada_trabalho','gasto_pc_educacao','exp_anos_estudo','nota_mat']
#Conversão dos campos do tipo objetos para categoricos

for i in df[lista].select_dtypes(include=['object']):

    df[i] = df[i].astype('category').cat.codes
#Substitui os campos nulos por -2

df.fillna(-2,inplace = True)
#Separação das bases após todos os tratamentos

ts = df[df['nota_mat']==-2]

df = df[df['nota_mat']!=-2]
#Separação da base em treino e validação, o train_test_split realiza a divisão aleatória dos dados do dataset

from sklearn.model_selection import train_test_split

train, valid = train_test_split(df[lista], random_state=42)

train.shape, valid.shape
#Criação do processo de floresta de decisão, onde são criados varias arvores de decisão e criado uma média para chegar ao melhor resultado

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_estimators=1000) #,, min_samples_leaf=1000 min_samples_leaf=650

feats = [c for c in df[lista].columns if c not in ['nota_mat']]

rf.fit(train[feats], train['nota_mat'])
#Realização da validação do modelo criado

from sklearn.metrics import mean_squared_error

mean_squared_error(rf.predict(valid[feats]), valid['nota_mat'])**(1/2)

valid_preds = rf.predict(valid[feats])

mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
#Realiza a predição das notas de matemática

ts['nota_mat'] = rf.predict(ts[feats])
#Geração do arquivo para envio

ts[['codigo_mun','nota_mat']].to_csv('lucio.csv',index=False)