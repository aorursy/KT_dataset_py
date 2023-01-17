# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# import re

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dftreino = pd.read_csv('../input/train.csv')

dfteste = pd.read_csv('../input/test.csv')

dftotal = dfteste.append(dftreino)
#df.head().T

#test.head().T

# train.head().T # nota_mat

dftotal.info()
dftotal['populacao']=dftotal['populacao'].str.replace('(','').str.replace(')',

               '').str.replace(',','').str.replace('.','').astype(dtype=np.int)


dftotal['area']=dftotal['area'].str.replace(',','').astype(float)

dftotal['densidade_dem']=dftotal['densidade_dem'].str.replace(',','').astype(float)

dftotal['cat_porte'] = dftotal['porte'].astype('category').cat.codes

dftotal['cat_regiao'] = dftotal['regiao'].astype('category').cat.codes

dftotal['cat_estado'] = dftotal['estado'].astype('category').cat.codes
dftotal['gasto_pc_educacao'].fillna(dftotal['gasto_pc_educacao'].mean(), inplace=True)

dftotal['perc_pop_econ_ativa'].fillna(dftotal['perc_pop_econ_ativa'].mean(), inplace=True)

# df['exp_vida'].fillna(df['exp_vida'].mean(), inplace=True)
feats = [c for c in dftotal.columns if c not in ['codigo_mun', 'comissionados_por_servidor','nota_mat', 'densidade_dem', 

                                            'participacao_transf_receita', 'servidores',  

                                            'gasto_pc_saude', 'hab_p_medico', 'exp_vida',  

                                            'exp_anos_estudo', 'regiao', 'estado', 'porte', 'municipio']]



# 'densidade_dem', 'gasto_pc_educacao', 'perc_pop_econ_ativa',
dftotal.shape, dftreino.shape, dfteste.shape
dftreino=dftotal[~dftotal.nota_mat.isnull()]

dfsubmissao=dftotal[dftotal.nota_mat.isnull()]
from sklearn.model_selection import train_test_split

dftreino2, dfteste = train_test_split(dftreino, test_size=0.20, random_state=42)

dftreino2, dfvalida = train_test_split(dftreino2, test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, max_depth=4)
# for c in df.select_dtypes('object').columns:

#    print(c)

#    if df[c].str.contains(',').max():

#        print(c)


rf.fit(dftreino2[feats], dftreino2['nota_mat'])

predicao = rf.predict(dfvalida[feats])
from sklearn.metrics import accuracy_score

accuracy_score(dfvalida['nota_mat'], predicao)
accuracy_score(dfteste['nota_mat'], rf.predict(dfteste[feats]))
dfsubmissao['nota_mat']=rf.predict(dfsubmissao[feats])

dfsubmissao[['codigo_mun','nota_mat']].to_csv('dfresultado.csv', index=False)

pd.Series(rf.feature_importances_,index=feats).sort_values().plot.barh()
# Treinando o modelo

# rf.fit(train[feats], train['nota_mat'])