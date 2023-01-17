import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import matplotlib

from sklearn.metrics import accuracy_score

import os

print(os.listdir("../input"))


df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')


df_train.shape, df_test.shape
df_train.dtypes, df_test.dtypes
df_train.info(), df_test.info()
df_train.describe()
df_train.head()


df_train = df_train.append(df_test)
df_train.shape
df_train['codigo_mun'] = df_train['codigo_mun'].str.replace('ID_ID_','').astype(int)
df_train = pd.concat([df_train, pd.get_dummies(df_train['porte'], prefix='porte').iloc[:, :-1]], 

                   axis =1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['regiao'], prefix='regiao').iloc[:, :-1]], 

                   axis =1)
#Convertendo estado para numero

df_train['estado'] = df_train['estado'].astype('category').cat.codes
#Convertendo municipio para numero

df_train['municipio'] = df_train['municipio'].astype('category').cat.codes
df_train.head()
#apagando todos NaNs

#df_train.dropna(inplace=True)
df_train.update(df_train['exp_anos_estudo'].fillna(df_train['exp_anos_estudo'].mean()))

#df_train.update(df_train['exp_anos_estudo'].fillna(0))

#df_train.update(df_train['exp_anos_estudo'].fillna(df_train['exp_anos_estudo'].median()))
df_train.update(df_train['exp_vida'].fillna(df_train['exp_vida'].mean()))

#df_train.update(df_train['exp_vida'].fillna(0))

#df_train.update(df_train['exp_vida'].fillna(df_train['exp_vida'].median()))
df_train.update(df_train['idhm'].fillna(df_train['idhm'].mean()))

#df_train.update(df_train['idhm'].fillna(0))

#df_train.update(df_train['idhm'].fillna(df_train['idhm'].median()))
df_train.update(df_train['indice_governanca'].fillna(df_train['indice_governanca'].mean()))

#df_train.update(df_train['indice_governanca'].fillna(0))

#df_train.update(df_train['indice_governanca'].fillna(df_train['indice_governanca'].median()))
#df_train.update(df_train['participacao_transf_receita'].fillna(df_train['participacao_transf_receita'].mean()))

#df_train.update(df_train['participacao_transf_receita'].fillna(0))

df_train.update(df_train['participacao_transf_receita'].fillna(df_train['participacao_transf_receita'].median()))
df_train.update(df_train['perc_pop_econ_ativa'].fillna(df_train['perc_pop_econ_ativa'].mean()))

#df_train.update(df_train['perc_pop_econ_ativa'].fillna(0))

#df_train.update(df_train['perc_pop_econ_ativa'].fillna(df_train['perc_pop_econ_ativa'].median()))
df_train.update(df_train['gasto_pc_saude'].fillna(df_train['gasto_pc_saude'].mean()))

#df_train.update(df_train['gasto_pc_saude'].fillna(0))

#df_train.update(df_train['gasto_pc_saude'].fillna(df_train['gasto_pc_saude'].median()))
df_train.info()
#quem não tiver nota_mat nulla vai para o DF de treino

train_raw = df_train[~df_train['nota_mat'].isnull()]
#quem tiver nota_mat nulla vai para o DF de teste

df_test = df_train[df_train['nota_mat'].isnull()]
#atribuindo 

df_train = train_raw
df_train.shape, df_test.shape
df_train.corr().nota_mat
df_train['nota_mat'] = np.log(df_train['nota_mat'])
df_train['nota_mat'].head()
from sklearn.model_selection import train_test_split

#separando em train e valid

train, valid = train_test_split(df_train, random_state=42)
train.shape, valid.shape
#importando algoritmos

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
#Colunas a serem removidas do DF

removed_cols = ['populacao', 'pib', 'hab_p_medico', 'gasto_pc_educacao', 'regiao', 'nota_mat', 'Unnamed: 0', 'area', 'capital', 'comissionados', 'comissionados_por_servidor', 'densidade_dem','municipio',

                'porte', 'ranking_igm', 'servidores', 'taxa_empreendedorismo']

feats = [c for c in df_train.columns if c not in removed_cols]
feats
from sklearn.metrics import mean_squared_error
def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds)**(1/2)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
#Criando score para cada um dos algoritmos

scores = []

for name, model in models.items():

    score = run_model(model, train, valid, feats, 'nota_mat')

    scores.append(score)

    print(name+':', score)
#Plotando scores

pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()

#Criando coluna preds com a média das notas de matematica

train['preds'] = train['nota_mat'].mean()
train['preds']
train.shape, train['nota_mat'].mean()

mean_squared_error(train['nota_mat'], train['preds'])
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf.fit(train[feats], train['nota_mat'])

#rodando o modelo para o df de treino com as colunas que não foram removidas

train_preds = rf.predict(train[feats])

train_preds
mean_squared_error(train['nota_mat'], train_preds)**(1/2)
#rodando o modelo para o df de validação com as colunas que não foram removidas

valid_preds = rf.predict(valid[feats])

mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
df_test['nota_mat'] = np.exp(rf.predict(df_test[feats]))
#Plotando as variáveis que mais influenciam o modelo

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()

df_test[['codigo_mun','nota_mat']].to_csv('rf.csv', index=False)