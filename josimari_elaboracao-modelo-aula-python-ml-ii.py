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
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df.shape, test.shape
df.describe()
test.describe()
import seaborn as sns
sns.pairplot(x_vars='exp_anos_estudo', y_vars='nota_mat', data=df)
df.groupby('estado')['nota_mat'].mean().plot.bar()
df = df.append(test, sort=False)
df_raw = df.copy()
df.shape
df.head().T
df.dtypes
df.codigo_mun
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
df.codigo_mun
df.area
df.area.str.replace(',','')
df.area = df.area.str.replace(',','').astype(float)
df.densidade_dem
df.densidade_dem.str.replace(',','')
df.densidade_dem = df.densidade_dem.str.replace(',','').astype(float)
df.servidores = df.servidores.values.astype(np.int64)
df.comissionados = df.comissionados.values.astype(np.int64)
df.comissionados_por_servidor = df.comissionados_por_servidor.astype('category')
for c in df.columns:

    if df[c].dtype == 'object':

        df[c] = df[c].astype('category').cat.codes
df.info()
df.isnull()
df.isnull().sum()
df['densidade_dem'].fillna(df['densidade_dem'].min(), inplace=True)
df['participacao_transf_receita'].fillna(df['participacao_transf_receita'].mean(), inplace=True)
df['perc_pop_econ_ativa'].fillna(np.log(df['perc_pop_econ_ativa']).mean(), inplace=True)
df['gasto_pc_saude'].fillna(np.log(df['gasto_pc_saude']).mean(), inplace=True)
df['hab_p_medico'].fillna(df['hab_p_medico'].mean(), inplace=True)
df['exp_vida'].fillna(df['exp_vida'].mean(), inplace=True)
df['gasto_pc_educacao'].fillna(np.log(df['gasto_pc_educacao']).mean(), inplace=True)
df['exp_anos_estudo'].fillna(df['exp_anos_estudo'].mean(), inplace=True)
df['idhm'].fillna(df['idhm'].mean(), inplace=True)
df['indice_governanca'].fillna(df['indice_governanca'].mean(), inplace=True)
df.isnull().sum()
df = df.drop(['participacao_transf_receita', 'servidores', 'comissionados_por_servidor', 'gasto_pc_saude', 'hab_p_medico', 'exp_vida', 'exp_anos_estudo', 'idhm', 'indice_governanca'], axis=1)
df.head()
df.isnull().sum()
df, test = df[~df['nota_mat'].isnull()], df[df['nota_mat'].isnull()]
df.shape, test.shape
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df.columns if c not in ['nota_mat']]
rf.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
test['nota_mat'] = rf.predict(test[feats])
test[['codigo_mun','nota_mat']].to_csv('Josimari_1831133057.csv', index=False)