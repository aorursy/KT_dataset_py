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
df = pd.read_csv("../input/train.csv")
df.describe()
df.head().T
test = pd.read_csv("../input/test.csv")
test.head().T
df.shape
test.shape
df.shape
df.head().T
df.isnull().sum()
df2 = df

test2= test

df = pd.concat([df, test])

df['estado'] = df['estado'].astype('category').cat.codes.astype(int)

df['municipio'] = df['municipio'].astype('category').cat.codes.astype(int)

df['regiao'] = df['regiao'].astype('category').cat.codes.astype(int)

df['porte'] = df['regiao'].astype('category').cat.codes.astype(int)

df.info()
df.info()
df['populacao'] = df['populacao'].str.replace(',','').str.replace('(','').str.replace(')','').astype(float)

df['area'] = df['area'].str.replace(',','').str.replace('(','').str.replace(')','').astype(float)

df['densidade_dem'] = df['densidade_dem'].str.replace(',','').astype(float)
df.info()
df['comissionados_por_servidor']= df['comissionados_por_servidor'].str.replace('#DIV/0!','0')

df['comissionados_por_servidor']= (df['comissionados_por_servidor'].str.replace('%','').astype(float))/100
densidade_media = df['densidade_dem'].mean()
densidade_media
df.update(df['densidade_dem'].fillna(densidade_media))
exp_media = df['exp_vida'].mean()
df.update(df['exp_vida'].fillna(exp_media))
exp_anos_media = df['exp_anos_estudo'].mean()

df.update(df['exp_anos_estudo'].fillna(exp_anos_media))

gasto_med = df['gasto_pc_educacao'].mean()

df.update(df['gasto_pc_educacao'].fillna(gasto_med))
saude_med =  df['gasto_pc_saude'].mean()

df.update(df['gasto_pc_saude'].fillna(saude_med))
hab_med = df['hab_p_medico'].mean() 



df.update(df['hab_p_medico'].fillna(hab_med))
part_med = df['participacao_transf_receita'].mean()

df.update(df['participacao_transf_receita'].fillna(part_med))
serv_med = df['servidores'].mean()

df.update(df['servidores'].fillna(serv_med))
perc_med = df['perc_pop_econ_ativa'].mean() 

df.update(df['perc_pop_econ_ativa'].fillna(perc_med))
df.info()
df.isna().sum()
test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
df.isna().sum()
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape, test.shape
test = test.drop('nota_mat',axis=1)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(

n_estimators=100, 

min_samples_split=5, 

max_depth=12,

random_state=42,

n_jobs=-1

)
removed_cols = ['nota_mat','capital']
feats = [c for c in df.columns if c not in removed_cols]
feats
rf.fit(train[feats],train['nota_mat'])
pd.Series(rf.feature_importances_,index=feats)
preds = rf.predict(valid[feats])
from sklearn.metrics import accuracy_score
accuracy_score(valid['nota_mat'],preds)
preds_test = rf.predict(test[feats])
preds_test
test['nota_mat'] = preds_test
test[['codigo_mun', 'nota_mat']].to_csv('predicao_lv.csv', index=False)