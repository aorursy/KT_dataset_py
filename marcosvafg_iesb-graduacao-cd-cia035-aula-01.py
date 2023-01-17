# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Carregando os dados

df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')



df.shape
# Verificando os dados

df.info()
# Olhando os dados

df.head()
# Olhando a coluna city

df['city'].value_counts()
# Verificando a distribuição do valor do aluguel por cidade

df[df['city'] == 'São Paulo']['rent amount (R$)'].plot.hist(bins=50)
# Verificando os 10 maiores valores de aluguel em São Paulo

df[df['city'] == 'São Paulo'].nlargest(10, 'rent amount (R$)')
# Verificando os 10 maiores valores de aluguel em Porto Alegre

df[df['city'] == 'Porto Alegre'].nlargest(10, 'rent amount (R$)')
# Verificando os 10 maiores valores de aluguel em Belo Horizonte

df[df['city'] == 'Belo Horizonte'].nlargest(10, 'rent amount (R$)')
# Distribuição dos dados sobre quartos

df['rooms'].value_counts()
# Distribuição dos dados sobre banheiros

df['bathroom'].value_counts()
# Distribuição dos dados sobre garagem

df['parking spaces'].value_counts()
# Verificando o imovel com 13 quartos

df[df['rooms'] == 13]
# Olhando a correlação das variáveis

import seaborn as sns



sns.heatmap(df.corr(), annot=True)
# Tratamento de dados



# Convertendo a coluna floor

df[df['floor'] == '-'] = '0'

df['floor'] = df['floor'].astype(int)



# Convertendo as colunas categóricas

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
# Importando as bibliotecas

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
# Divisão dos dados

train, test = train_test_split(df)



train.shape, test.shape
# Usando Random Forest

# Sem remover a coluna Rent Amount



# Features

feats = [col for col in df.columns if col not in ['total (R$)']]



# Instanciar o modelo

rf1 = RandomForestRegressor(random_state=42, n_estimators=200, oob_score=True)



# Treinar o modelo

rf1.fit(train[feats], train['total (R$)'])



# Previsões usando o modelo

preds1 = rf1.predict(test[feats])



# Avaliar o desempenho

mean_squared_error(test['total (R$)'], preds1)
# Usando Random Forest

# Removendo a coluna Rent Amount



# Features

feats = [col for col in df.columns if col not in ['rent amount (R$)', 'total (R$)']]



# Instanciar o modelo

rf2 = RandomForestRegressor(random_state=42, n_estimators=200, oob_score=True)



# Treinar o modelo

rf2.fit(train[feats], train['total (R$)'])



# Previsões usando o modelo

preds2 = rf2.predict(test[feats])



# Avaliar o desempenho

mean_squared_error(test['total (R$)'], preds2)
# Atividade para 17/09/2020



# - Melhorar tratamento dos dados

# - Possibilidade de feature engineering

# - Utilização de outros algoritimos

# - OBJETIVO: Melhorar o desempenho da previsão