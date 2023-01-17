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
# Carregando os dados

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

train.head().T
test.head().T
submission.head().T
test.info()
train.info()
train['populacao'] = train['populacao'].str.replace(",", "", regex=False)

train['populacao'] = train['populacao'].str.replace("(2)", "", regex=False)

train['populacao'] = train['populacao'].str.replace("()", "", regex=False)

train['populacao'] = train['populacao'].str.replace("(1)", "", regex=False)

train['populacao'] = train['populacao'].astype(float)
test['populacao'] = test['populacao'].str.replace(",", "", regex=False)

test['populacao'] = test['populacao'].str.replace("(2)", "", regex=False)

test['populacao'] = test['populacao'].astype(float)
train['area'] = train['area'].str.replace(",", "", regex=False)

train['area'] = train['area'].astype(float)
test['area'] = test['area'].str.replace(",", "", regex=False)

test['area'] = test['area'].astype(float)
train['densidade_dem'] = train['densidade_dem'].str.replace(",", "", regex=False)

train['densidade_dem'] = train['densidade_dem'].astype(float)
test['densidade_dem'] = test['densidade_dem'].str.replace(",", "", regex=False)

test['densidade_dem'] = test['densidade_dem'].astype(float)
train['servidores'] = np.where(train['servidores'].isna(), round(train['populacao']*train.servidores.mean() / train.populacao.mean(), 0)

                               , train['servidores'])
test['servidores'] = np.where(test['servidores'].isna(), round(test['populacao']*test.servidores.mean() / test.populacao.mean(), 0)

                               , test['servidores'])
test['comissionados_por_servidor'] = test['comissionados'] / test['servidores']
train['comissionados_por_servidor'] = train['comissionados'] / train['servidores']
df = test.copy()
df = df.append(train, sort=False)
df.info()
for col in df.columns:

    if df[col].dtype=='object':

        df[col] = df[col].astype('category').cat.codes
df['densidade_dem'] = np.where(df['densidade_dem'].isna(), round(df['populacao'] / df['area'], 0)

                               , df['densidade_dem'])
df['participacao_transf_receita'] = np.where(df['participacao_transf_receita'].isna()

                                             , df.groupby('porte')['participacao_transf_receita'].transform('mean')

                                             , df['participacao_transf_receita'])
df['perc_pop_econ_ativa'] = np.where(df['perc_pop_econ_ativa'].isna()

                                             , df.groupby('porte')['perc_pop_econ_ativa'].transform('mean')

                                             , df['perc_pop_econ_ativa'])
df['gasto_pc_saude'] = np.where(df['gasto_pc_saude'].isna()

                                             , df.groupby('porte')['gasto_pc_saude'].transform('mean')

                                             , df['gasto_pc_saude'])

df['hab_p_medico'] = np.where(df['hab_p_medico'].isna()

                                             , df.groupby('porte')['hab_p_medico'].transform('mean')

                                             , df['hab_p_medico'])

df['exp_vida'] = np.where(df['exp_vida'].isna()

                                             , df.groupby('porte')['exp_vida'].transform('mean')

                                             , df['exp_vida'])
df['gasto_pc_educacao'] = np.where(df['gasto_pc_educacao'].isna()

                                             , df.groupby('porte')['gasto_pc_educacao'].transform('mean')

                                             , df['gasto_pc_educacao'])

df['exp_anos_estudo'] = np.where(df['exp_anos_estudo'].isna()

                                             , df.groupby('porte')['exp_anos_estudo'].transform('mean')

                                             , df['exp_anos_estudo'])
df.info()
train = df[~df['nota_mat'].isnull()]

test = df[df['nota_mat'].isnull()]
from sklearn.model_selection import train_test_split

treino, valid = train_test_split(train, random_state = 42)
treino.shape, valid.shape
removed_columns = ['nota_mat', 'municipio']

feats = [c for c in train.columns if c not in removed_columns]
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
models= {'RandomForest': RandomForestRegressor(random_state=42),

         'ExtraTrees': ExtraTreesRegressor(random_state=42),

         'GradientBoosting': GradientBoostingRegressor(random_state=42),

         'DecisionTree': DecisionTreeRegressor(random_state=42),

         'AdaBoost': AdaBoostRegressor(random_state=42),

         'KNM 1': KNeighborsRegressor(n_neighbors=1),

         'KNM 3': KNeighborsRegressor(n_neighbors=3),

         'KNM 11': KNeighborsRegressor(n_neighbors=11),

         'SVR': SVR(),

         'Linear Regression': LinearRegression()}
from sklearn.metrics import mean_squared_error
def run_model(model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds) ** (1/2)
scores = []

for name, model in models.items():

    score = run_model(model, train.fillna(-1), valid.fillna(-1), feats, 'nota_mat')

    scores.append(score)

    print(name, ':', score)
pd.Series(scores, index=models.keys()).sort_values(ascending=False).plot.barh()