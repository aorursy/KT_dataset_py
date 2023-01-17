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
df_x = pd.read_csv("../input/train.csv")
df = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
df.shape, test.shape
df = df.append(test,ignore_index=True)
df.head()
df.shape
df.dtypes
df.head()
df.info()
df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_','').astype(int)
# df[df['area']=='5,126.72'] -- não consigo alterar o tipo deste atributo para float. Preciso retirar retirar a vírgula

#df['area'].replace([","],"",inplace=False)

df['area'] = df['area'].str.replace(',','')
df['area'] = df.area.astype(float)
df.info()
#df[df['densidade_dem']=='1,973.60'] #-- não consigo alterar o tipo deste atributo para float. Preciso retirar retirar a vírgula

df['densidade_dem'] = df['densidade_dem'].str.replace(',','')

df['densidade_dem'] = df.densidade_dem.astype(float)
df.info()
df[['servidores', 'comissionados','comissionados_por_servidor','estado','porte']]
#df.groupby(['estado','porte']).servidores.mean()

df.groupby(['estado','porte']).servidores.median()
#df['servidores'].fillna(df.groupby(['estado','porte'])['servidores'].transform('mean'),inplace=True)

df['servidores'].fillna(df.groupby(['estado','porte'])['servidores'].transform('mean'),inplace=True)
df['servidores'] = round(df['servidores'],0)
df[['servidores', 'comissionados','comissionados_por_servidor','estado','porte']]
df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('%','')

#df['comissionados_por_servidor'] = df.comissionados_por_servidor.astype(float)
for i in df.index:

    if df.at[i,'comissionados_por_servidor'] == '#DIV/0!':

        df.at[i,'comissionados_por_servidor'] = df.at[i,'comissionados'] * 100 / df.at[i,'servidores']



df[['servidores', 'comissionados','comissionados_por_servidor','estado','porte']]
df['comissionados_por_servidor'] = df.comissionados_por_servidor.astype(float)
df.info()
df[df.perc_pop_econ_ativa.isnull()]
df['perc_pop_econ_ativa'].fillna(df.groupby(['estado','porte'])['perc_pop_econ_ativa'].transform('mean'),inplace=True)
df.info()
df.head()
df[df.densidade_dem.isnull()]
# df[(df['regiao'] == 'SUL') & (df['porte'] == 'Pequeno porte 1')].head()

#df['densidade_dem'].fillna(df.groupby(['estado','porte'])['densidade_dem'].transform('median'),inplace=True)

df['densidade_dem'].fillna(df.groupby(['estado','porte'])['densidade_dem'].transform('mean'),inplace=True)
df[df.densidade_dem.isnull()]
df.info()
df['participacao_transf_receita'].hist()
df[df.participacao_transf_receita.isnull()]
df.head()
#df.groupby(['estado','porte']).participacao_transf_receita.median()

df.groupby(['estado','porte']).participacao_transf_receita.mean()
#df['participacao_transf_receita'].fillna(df.groupby(['estado','porte'])['participacao_transf_receita'].transform('median'),inplace=True)

df['participacao_transf_receita'].fillna(df.groupby(['estado','porte'])['participacao_transf_receita'].transform('mean'),inplace=True)
df.info()
# df.groupby(['regiao']).participacao_transf_receita.median()

df.groupby(['regiao']).participacao_transf_receita.mean()
# trantando o restante dos valores null

#df['participacao_transf_receita'].fillna(df.groupby(['regiao'])['participacao_transf_receita'].transform('median'),inplace=True)

df['participacao_transf_receita'].fillna(df.groupby(['regiao'])['participacao_transf_receita'].transform('mean'),inplace=True)
df.info()
df['gasto_pc_saude'].hist()
#df.groupby(['regiao']).gasto_pc_saude.median()

df.groupby(['regiao']).gasto_pc_saude.mean()
#df.groupby(['estado', 'porte']).gasto_pc_saude.median()

df.groupby(['estado', 'porte']).gasto_pc_saude.mean()
# df['gasto_pc_saude'].fillna(df.groupby(['estado','porte'])['gasto_pc_saude'].transform('median'),inplace=True)

df['gasto_pc_saude'].fillna(df.groupby(['estado','porte'])['gasto_pc_saude'].transform('mean'),inplace=True)
df.info()
df[df.gasto_pc_saude.isnull()]
media_gasto_saude_norte = df['gasto_pc_saude'][(df['regiao'] == 'NORTE') & (df['porte'] == 'Pequeno porte 1')].mean()
df['gasto_pc_saude'].fillna(media_gasto_saude_norte,inplace=True)
df.info()
df['hab_p_medico'].hist()
df[df.hab_p_medico.isnull()]
#df.groupby(['estado', 'porte']).hab_p_medico.median()

df.groupby(['estado', 'porte']).hab_p_medico.mean()
#df['hab_p_medico'].fillna(df.groupby(['estado','porte'])['hab_p_medico'].transform('median'),inplace=True)

df['hab_p_medico'].fillna(df.groupby(['estado','porte'])['hab_p_medico'].transform('mean'),inplace=True)
df.info()
df['exp_vida'].hist()
df[df.exp_vida.isnull()]
# df['exp_vida'].fillna(df.groupby(['estado','porte'])['exp_vida'].transform('median'),inplace=True)

df['exp_vida'].fillna(df.groupby(['estado','porte'])['exp_vida'].transform('mean'),inplace=True)
df.info()
df['gasto_pc_educacao'].hist()
df[df.gasto_pc_educacao.isnull()]
#df.groupby(['estado', 'porte']).gasto_pc_educacao.median()

df.groupby(['estado', 'porte']).gasto_pc_educacao.mean()
#df['gasto_pc_educacao'].fillna(df.groupby(['estado','porte'])['gasto_pc_educacao'].transform('median'),inplace=True)

df['gasto_pc_educacao'].fillna(df.groupby(['estado','porte'])['gasto_pc_educacao'].transform('mean'),inplace=True)
df.info()
media_gasto_educacao_norte = df['gasto_pc_educacao'][(df['regiao'] == 'NORTE') & (df['porte'] == 'Pequeno porte 1')].mean()
df['gasto_pc_educacao'].fillna(media_gasto_educacao_norte,inplace=True)
df.info()
df['exp_anos_estudo'].hist()
df[df.exp_anos_estudo.isnull()]
df.head()
#df.groupby(['estado','porte']).exp_anos_estudo.median()

df.groupby(['estado','porte']).exp_anos_estudo.mean()
#df['exp_anos_estudo'].fillna(df.groupby(['estado','porte'])['exp_anos_estudo'].transform('median'),inplace=True)

df['exp_anos_estudo'].fillna(df.groupby(['estado','porte'])['exp_anos_estudo'].transform('mean'),inplace=True)
df[(df['regiao'] == 'SUL') & (df['municipio'] == 'BALNEARIO RINCAO')]
df.info()
df['idhm'].hist()
df[df['idhm'].isnull()]
df.head()
df.groupby(['estado','porte']).idhm.mean()
df['idhm'].fillna(df.groupby(['estado','porte'])['idhm'].transform('mean'),inplace=True)
df.info()
df.head(100)
df[df['indice_governanca'] == 0.598]
# df.groupby(['estado','porte']).indice_governanca.median()

df.groupby(['estado','porte']).indice_governanca.mean()
df.groupby(['estado']).indice_governanca.mean()
#df['indice_governanca'].fillna(df.groupby(['estado'])['indice_governanca'].transform('median'),inplace=True)

df['indice_governanca'].fillna(df.groupby(['estado'])['indice_governanca'].transform('mean'),inplace=True)
df.info()
df['ranking_igm'] = df['ranking_igm'].str.replace('º','')
df[['porte','populacao','area','densidade_dem','ranking_igm','indice_governanca']][df['ranking_igm'].notnull()].sort_values('ranking_igm',ascending=True)
df[['Unnamed: 0','porte','populacao','area','densidade_dem','ranking_igm','indice_governanca']].sort_values('ranking_igm',ascending=True)
df['Unnamed: 0']
df['ranking_igm'].fillna('X',inplace=True)
df.info()


for i in df.index:

    if df.at[i,'ranking_igm'] == 'X':

        df.at[i,'ranking_igm'] = df.at[i,'Unnamed: 0'] + 1



#df[df.ranking_igm.isnull()] = df['Unnamed: 0'] + 1    
df.groupby('ranking_igm').ranking_igm.count()
df.info()
df['ranking_igm'] = df.area.astype(int)
df.info()
dm_regiao = pd.get_dummies(df['regiao'], prefix='reg')
dm_regiao.head()
dm_regiao.info()
dm_regiao[['reg_CENTRO-OESTE','reg_NORDESTE','reg_NORTE','reg_SUDESTE','reg_SUL']]=dm_regiao[['reg_CENTRO-OESTE','reg_NORDESTE','reg_NORTE','reg_SUDESTE','reg_SUL']].astype(int)
dm_regiao.info()
#Realizando a junção

#df = df.append(dm_regiao,axis=0)

df = pd.concat([df, dm_regiao], axis = 1)
df.info()
dm_porte = pd.get_dummies(df['porte'], prefix='p')
dm_porte.head()
dm_porte[['p_Grande porte','p_Médio porte','p_Pequeno porte 1','p_Pequeno porte 2']]=dm_porte[['p_Grande porte','p_Médio porte','p_Pequeno porte 1','p_Pequeno porte 2']].astype(int)
dm_porte.info()
df = pd.concat([df, dm_porte], axis = 1)
df.shape
df.info()
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
treino = df[~df['nota_mat'].isnull()]

test   = df[df['nota_mat'].isnull()]
treino.shape, test.shape
treino.info()
treino.corr().nota_mat
df.corr().style.background_gradient()
treino['nota_mat'] = np.log(treino['nota_mat'])
treino['nota_mat'].head()
from sklearn.model_selection import train_test_split
df_treino, df_valid = train_test_split(treino, random_state=42)
df_treino.shape, df_valid.shape
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
models = {'RandomForest': RandomForestRegressor(random_state=42),

         'ExtraTrees': ExtraTreesRegressor(random_state=42),

         'GBM': GradientBoostingRegressor(random_state=42),

         'DecisionTree': DecisionTreeRegressor(random_state=42),

         'AdaBoost': AdaBoostRegressor(random_state=42),

         'KNN 1': KNeighborsRegressor(n_neighbors=1),

         'KNN 3': KNeighborsRegressor(n_neighbors=3),

         'KNN 11': KNeighborsRegressor(n_neighbors=11),

         'SVR': SVR(),

         'Linear Regression': LinearRegression()}
removed_cols = ['nota_mat', 'Unnamed: 0', 'area', 'capital', 'comissionados', 'comissionados_por_servidor', 'densidade_dem', 'municipio',

               'regiao', 'porte', 'ranking_igm', 'taxa_empreendedorismo', 'hab_p_medico']
feats = [c for c in df_treino.columns if c not in removed_cols]
from sklearn.metrics import mean_squared_error
def run_model(model, df_treino, df_valid, feats, y_name):

    model.fit(df_treino[feats], df_treino[y_name])

    preds = model.predict(df_valid[feats])

    return mean_squared_error(df_valid[y_name], preds)**(1/2)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
scores = []

for name, model in models.items():

    score = run_model(model, df_treino, df_valid, feats, 'nota_mat')

    scores.append(score)

    print(name+':', score)
pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()
df_treino['preds'] = df_treino['nota_mat'].mean()
df_treino.shape, df_treino['nota_mat'].mean()
mean_squared_error(df_treino['nota_mat'], df_treino['preds'])
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(df_treino[feats], df_treino['nota_mat'])
train_preds = rf.predict(df_treino[feats])
train_preds
mean_squared_error(df_treino['nota_mat'], train_preds)**(1/2)
valid_preds = rf.predict(df_valid[feats])
mean_squared_error(df_valid['nota_mat'], valid_preds)**(1/2)
#test['nota_mat'] = np.exp(rf.predict(test[feats]))

test['nota_mat'] = np.exp(rf.predict(test[feats]))
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
test.head()
test[['codigo_mun','nota_mat']].to_csv('rf.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)
gb.fit(df_treino[feats], df_treino['nota_mat'])
train_preds = gb.predict(df_treino[feats])
train_preds
mean_squared_error(df_treino['nota_mat'], train_preds)**(1/2)
valid_preds = gb.predict(df_valid[feats])
mean_squared_error(df_valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = np.exp(gb.predict(test[feats]))
pd.Series(gb.feature_importances_, index=feats).sort_values().plot.barh()
test[['codigo_mun','nota_mat']].to_csv('gb.csv', index=False)
import matplotlib.pyplot as plt

import seaborn as sns
df.describe()
plt.figure(figsize=(10,5))

sns.boxplot(x='regiao', y='nota_mat', data=df_x, fliersize=2)
plt.figure(figsize=(10,5))

sns.scatterplot(x='exp_anos_estudo', y='nota_mat', data=df_x)
plt.figure(figsize=(10,5))

sns.scatterplot(x='indice_governanca', y='nota_mat', data=df_x)
nm_estado = pd.DataFrame(df_x.groupby(['regiao','estado'], as_index=False).nota_mat.mean())

plt.figure(figsize=(100,300))

sns.catplot(x='regiao', y='nota_mat', data=nm_estado, hue='estado', kind='bar', height=6, aspect=2)

plt.xticks(rotation=45)





help(sns.catplot)