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

### DICAS E INSTRUÇÕES ESTÃO EM: https://www.kaggle.com/c/iesb-2019/ OU bit.ly/iesb-2019
## REFERENCIAS: https://datawhatnow.com/feature-importance/
                # https://www.youtube.com/watch?v=WE67TSz-a7s
                # https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-
        
# http://minerandodados.com.br/index.php/2017/09/01/cafe-com-codigo-tratando-valores-faltantes-pandas-python/
# http://felipegalvao.com.br/blog/2016/02/29/manipulacao-de-dados-com-python-pandas/
# https://www.kaggle.com/joaoavf/kernels
        
        # https://www.kaggle.com/manoxis/iesb-python-e-pandas-aula-06
        # https://www.kaggle.com/manoxis/iesb-python-e-pandas-aula-05
        # GRÁFICOS: https://www.kaggle.com/manoxis/aula-03-python-e-pandas/edit
df = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")
df.shape, test.shape
df.describe().T
df.isnull().sum()
df.dtypes
df = df.append(test)
# ALTERAÇÃO DA COLUNA CHAVE DA AMOSTRA, RETIRANTO A STRING 'ID_ID_' E CONVERTENDO PARA NUMÉRICO
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_',''))
df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
# SHAPE PARA CONFIRMAR QUE OS DATA FRAMES AINDA ESTÃO JUNTOS
df.shape
df.isnull().sum()
# Transformar texto em número
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].astype('category').cat.codes
# como essa coluna não existia inicialmente no arquivo, utilizamos ela para filtrar para separar os dataframes novamente

test = df[df['nota_mat'].isnull()]
df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
df.min().min()
test.min().min()
test.fillna(-10, inplace = True)
df.fillna(-10, inplace = True)
df['nota_mat'].hist(bins=50)
# plotando um histograma para a columa nota matemática
df.groupby('estado')['nota_mat'].mean().plot.bar()
df.groupby('porte')['nota_mat'].describe()['mean'].sort_index(ascending=False).plot()
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
# se eu utilizar o parâmentro "... (df, random_state=42, test_size =.1)" divido a amostra com 90% para treino e 10% para testes.
# o padrão, sem parâmentro nenhum, divide em 75% treino e 25% para testes
removed_cols = ['municipio','Unnamed: 0']
feats = [c for c in df.columns if c not in removed_cols]
train, valid = train_test_split(df, random_state=42)
# se eu utilizar o parâmentro "... (df, random_state=42, test_size =.1)" divido a amostra com 90% para treino e 10% para testes.
# o padrão, sem parâmentro nenhum, divide em 75% treino e 25% para testes
train.shape, valid.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_estimators=1000, min_samples_leaf=30, verbose=3)
feats = [c for c in df.columns if c not in ['nota_mat']]
rf.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# abaixo enviaremos o modelo para o kaggle validar com uma vase de teste que não conhecemos
test['nota_mat'] = rf.predict(test[feats])
test[['codigo_mun','nota_mat']].to_csv('rf_rodrigoaragao.csv', index=False)