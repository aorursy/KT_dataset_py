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
import numpy as np

import pandas as pd

import os
os.listdir('../input')
df = pd.read_csv('../input/train.csv')

ts = pd.read_csv("../input/test.csv")
df.dtypes
import math
math.log(1000) - math.log(1100), math.log(10) - math.log(11)
test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
for c in df.columns:

 if df[c].dtype == 'object':

  df[c] = df[c].astype('category').cat.codes
df.min().min()



df.fillna(-1.0, inplace=True)
from sklearn.model_selection import train_test_split

df, valid = train_test_split(df, random_state=42)
removed_cols = ['municipio','codigo_mun']
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42, max_depth=2)
lista= ['estado','anos_estudo_empreendedor','nota_mat', 'municipio','codigo_mun','area','capital','codigo_mun',

        'comissionados','comissionados_por_servidor','densidade_dem','estado','exp_anos_estudo','exp_vida','gasto_pc_educacao',

        'gasto_pc_saude','hab_p_medico','idhm','indice_governanca','jornada_trabalho','municipio','nota_mat',

        'participacao_transf_receita', 'perc_pop_econ_ativa','pib','pib_pc','populacao','porte','ranking_igm',

        'regiao','servidores','taxa_empreendedorismo']
from sklearn.model_selection import train_test_split

train,valid = train_test_split(df,random_state=42)

train.shape,valid.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42,n_estimators=100)

feats = [c for c in df[lista].columns if c not in ['nota_mat']]

rf.fit(train[feats], train['nota_mat'])
math.exp(2.78), math.exp(5.27)
from sklearn.metrics import mean_squared_error
df['preds'] = df['nota_mat'].mean()
df.shape, df['nota_mat'].mean()
from sklearn.ensemble import RandomForestRegressor
df_preds = rf.predict(valid[feats])
df_preds
mean_squared_error(valid['nota_mat'], df_preds)**(1/2)
valid_preds=rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
import seaborn as sns

sns.distplot(pd.Series(rf.feature_importances_, index=feats).sort_values())

import matplotlib.pyplot as plt

plt.legend()
ts.fillna(-1.0, inplace=True)
ts.columns
ts['codigo_mun'] = ts['codigo_mun'].values.astype('int64')
for c in ts.columns:

 if ts[c].dtype == 'object':

  ts[c] = df[c].astype('category').cat.codes
ts.fillna(-1.0, inplace=True)
ts['nota_mat']=rf.predict(ts[feats])
ts[['codigo_mun','nota_mat']].to_csv('trabalho.csv', index=False)
ts[['codigo_mun','nota_mat']]