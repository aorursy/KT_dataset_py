# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df2 = pd.read_csv("../input/train.csv", parse_dates=[0])

test = pd.read_csv("../input/test.csv", parse_dates=[0])
df3 = df2.copy()

test3 = test.copy()
df2.shape, test.shape
#df = df2.copy()

df = df2.append(test, sort=False)
df.shape
df.head()
df.info()
df['populacao']= df['populacao'].str.replace(',','', regex=False)

df['populacao']= df['populacao'].str.replace('(2)','', regex=False)

df['populacao']= df['populacao'].str.replace('(1)','', regex=False)

df['populacao']= df['populacao'].str.replace('()','', regex=False)

df['populacao']= df.populacao.astype(float)
df['area']= df['area'].str.replace(',','', regex=False)

df['area']= df.area.astype(float)
df['densidade_dem']= df['densidade_dem'].str.replace(',','', regex=False)

df['densidade_dem']= df.densidade_dem.astype(float)
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
col_nan=df.isnull().any(axis=0)

col_nan
df['servidores']= np.where(df['servidores'].isna(), round(df['populacao']*df.servidores.mean()/df.populacao.mean(), 0),

                            df['servidores'])
df['gasto_pc_educacao'] = np.where(df['gasto_pc_educacao'].isna(), round(df['populacao']*df.gasto_pc_educacao.mean()/df.populacao.mean(), 2),

                            df['gasto_pc_educacao'])
df['comissionados_por_servidor']= df['comissionados']/df['servidores']
df['densidade_dem'] = np.where(df['densidade_dem'].isna(), round(df['populacao']/df['area'], 1),

                            df['densidade_dem'])
df['hab_p_medico'] = np.where(df['hab_p_medico'].isna(), -1, df['hab_p_medico'])

#df['hab_p_medico'] = np.where(df['hab_p_medico'].isna(), round(df['populacao']*df.hab_p_medico.mean()/df.populacao.mean(), 0),

#                             df['hab_p_medico'])
df['perc_pop_econ_ativa'] = np.where(df['perc_pop_econ_ativa'].isna(), -1, df['perc_pop_econ_ativa'])
df['participacao_transf_receita'] = np.where(df['participacao_transf_receita'].isna(), -1, df['participacao_transf_receita'])
df['gasto_pc_saude'] = np.where(df['gasto_pc_saude'].isna(), -1, df['gasto_pc_saude'])
#df['exp_vida'] = np.where(df['exp_vida'].isna(), df['exp_vida']== -1, df['exp_vida'])

df['exp_vida'] = np.where(df['exp_vida'].isna(), df['exp_vida'].mean(), df['exp_vida'])
#df['exp_anos_estudo'] = np.where(df['exp_anos_estudo'].isna(), df['exp_anos_estudo']== -1, df['exp_anos_estudo'])

df['exp_anos_estudo'] = np.where(df['exp_anos_estudo'].isna(), df['exp_anos_estudo'].mean(), df['exp_anos_estudo'])
col_nan=df.isnull().any(axis=0)

col_nan
df.head()
import seaborn as sns
df.corr()
sns.heatmap(df.corr())
test = df[df['nota_mat'].isnull()]
df = df[~df['nota_mat'].isnull()]
from sklearn.model_selection import train_test_split

train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
df.tail()
#df.sort_index(inplace=True)
removed_cols = ['nota_mat', 'municipio', 'hab_p_medico', 'capital', 'comissionados_por_servidor']
feats = [c for c in df.columns if c not in removed_cols]
feats
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)
rf.fit(train[feats], train['nota_mat'])
preds = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['nota_mat'], preds)
from sklearn.metrics import accuracy_score

accuracy_score(valid['nota_mat'], preds)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
train.head()
preds_test = rf.predict(test[feats])
test['nota_mat'] = preds_test
test[['codigo_mun', 'nota_mat']].to_csv('rf2.csv', index=False)