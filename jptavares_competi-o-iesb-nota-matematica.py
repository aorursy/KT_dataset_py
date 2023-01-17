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
df = df.append(test)
df.info()
test.info()
df.head().T
df.dtypes
# Transformando as variaveis categóricas

for col in ['regiao', 'estado', 'porte']:

    # Trabnsformar texto em categoria

    df[col] = df[col].astype('category')

    df[col] = df[col].cat.codes
df.sample(10)
df.dtypes
df['populacao'] = df['populacao'].str.replace(',','')

df['populacao'] = df['populacao'].str.replace('\(\)','')

df['populacao'] = df['populacao'].str.replace('\(1\)','')

df['populacao'] = df['populacao'].str.replace('\(2\)','')

df['populacao'] = df['populacao'].astype('float')
df['area'] = df['area'].str.replace(',','').astype('float')

df['densidade_dem'] = df['densidade_dem'].str.replace(',','').astype('float')
df.info()
df['in_densidade_dem'] = df['densidade_dem'].isna().astype('int')

df['densidade_dem'] = df['densidade_dem'].fillna(df['densidade_dem'].mean())
df['in_participacao_transf_receita'] = df['participacao_transf_receita'].isna().astype('int')

df['participacao_transf_receita'] = df['participacao_transf_receita'].fillna(df['participacao_transf_receita'].mean())
df['in_servidores'] = df['servidores'].isna().astype('int')

df['servidores'] = df['servidores'].fillna(df['servidores'].mean())
df['in_perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].isna().astype('int')

df['perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].fillna(df['perc_pop_econ_ativa'].mean())
df['in_gasto_pc_saude'] = df['gasto_pc_saude'].isna().astype('int')

df['gasto_pc_saude'] = df['gasto_pc_saude'].fillna(df['gasto_pc_saude'].mean())

df['in_hab_p_medico'] = df['hab_p_medico'].isna().astype('int')

df['hab_p_medico'] = df['hab_p_medico'].fillna(df['hab_p_medico'].mean())

df['in_exp_vida'] = df['exp_vida'].isna().astype('int')

df['exp_vida'] = df['exp_vida'].fillna(df['exp_vida'].mean())

df['in_gasto_pc_educacao'] = df['gasto_pc_educacao'].isna().astype('int')

df['gasto_pc_educacao'] = df['gasto_pc_educacao'].fillna(df['gasto_pc_educacao'].mean())

df['in_exp_anos_estudo'] = df['exp_anos_estudo'].isna().astype('int')

df['exp_anos_estudo'] = df['exp_anos_estudo'].fillna(df['exp_anos_estudo'].mean())
# Criando colunas númericas

df['div_gasto'] = df['gasto_pc_saude'] / df['pib']

#df['pop_pib'] = df['pib'] / df['populacao']

df['exp_pib'] = df['pib'] / df['exp_vida']

df['exp_pib_estudos'] = df['pib'] / df['exp_anos_estudo']
df.info()
df.head().T
test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
from sklearn.model_selection import train_test_split

train, valid = train_test_split(df, random_state=42)

train.shape, valid.shape
# Lista das colunas não utilizadas

removed_cols = ['municipio', 'comissionados_por_servidor', 'nota_mat']



# Lista das features

feats = [c for c in df.columns if c not in removed_cols]
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)
rf.fit(train[feats], train['nota_mat'])
preds = rf.predict(valid[feats])
from sklearn.metrics import accuracy_score



accuracy_score(valid['nota_mat'], preds)
# Utilizando validação cruzada

def cv(df, test, feats, y_name, k=10):

    preds, score, fis = [], [], []

    chunk = df.shape[0] // k

    for i in range(k):

        if i + 1 < k:

            valid = df.iloc[i*chunk: (i+1)*chunk]

            train = df.iloc[:i*chunk].append(df.iloc[(i+1)*chunk:])

        else:

            valid = df.iloc[i*chunk:]

            train = df.iloc[:i*chunk]

    

        rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=210, min_samples_leaf=5)

    

        rf.fit(train[feats], train[y_name])

        

        score.append(accuracy_score(valid[y_name], rf.predict(valid[feats])))

        

        preds.append(rf.predict(test[feats]))

        

        fis.append(rf.feature_importances_)

        

        print(i, 'OK')

    return score, preds, fis

score, preds, fis = cv(df, test, feats, 'nota_mat')
preds
preds[1].shape
score
pd.Series(score).mean()
final = pd.DataFrame({'codigo_mun': test['codigo_mun'], 'nota_mat': preds[2]})
final.to_csv('arquivopablo.csv', index=False)