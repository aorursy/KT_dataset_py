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
#Carregando os dados

test = pd.read_csv ('../input/test.csv',low_memory=False)

df = pd.read_csv ('../input/train.csv',low_memory=False)
#Criacao da coluna nota_mat no dataframe de teste. Atribuido valor padrão para permitir 

#a separação.

test['nota_mat'] = -2
df.T
#Juntando os dataframes

df = df.append(test)
df.T
#Verificando as colunas com null 

df.isnull().sum()
#Removendo o texto ID da coluna codigo_mun

df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
#Substituir todos os valores nulos por -10000. Valor -10000 usado após verificar o valor 

# mínimo de todas as variáveis, permintindo um recorte mais preciso.

#df[df.columns].min()

df.fillna(-10000, inplace=True)
#Transformando variaveis categorias, exceto codigo_mun para manter o tipo int64.

for col in df.columns:

    if df[col].dtype == 'object' and col != 'codigo_mun':

        df[col] = df[col].astype('category').cat.codes 
#Removendo colunas nao necessarias (inicialmente) ao modelo

removed_cols = ['Unnamed: 0', 'codigo_mun','municipio', 'nota_mat']

feats = [c for c in df.columns if c not in removed_cols]
#Separando os dataframes

test = df[df['nota_mat'] == -2]

df = df[~(df['nota_mat'] == -2)]
#Split em train e validação

from sklearn.model_selection import train_test_split

train, valid = train_test_split(df, test_size=0.3333, random_state=42)
train.shape, valid.shape
#Instancia o modelo

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)
#Executa o modelo

rf.fit(train[feats], train['nota_mat'])
#Calcula o erro quadrado

from sklearn.metrics import mean_squared_error

valid_preds = rf.predict(valid[feats])

mean_squared_error((valid['nota_mat']), valid_preds) ** (1/2)
#Avaliando a Importância das Features no Modelo

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize=(10,10))
#Removendo novas variaveis não relevantes ao modelo

removed_cols = ['exp_anos_estudo', 'densidade_dem','gasto_pc_saude', 'hab_p_medico','servidores',

                'area','comissionados','comissionados_por_servidor','indice_governanca',

                'ranking_igm','porte','capital']

feats = [c for c in df.columns if c not in removed_cols]
#Executa novamente o modelo

rf.fit(train[feats], train['nota_mat'])
#Calcula novamente o erro quadrado

valid_preds = rf.predict(valid[feats])

mean_squared_error((valid['nota_mat']), valid_preds) ** (1/2)
#Atribuindo o valor predito a variável nota_mat

test['nota_mat'] = rf.predict(valid[feats])
#Exporta o modelo pra .csv

test[['codigo_mun','nota_mat']].to_csv('rf_iesb_2019.csv', index=False)