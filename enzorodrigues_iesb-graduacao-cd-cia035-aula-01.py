import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Carregando os dados
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

df.shape
# Verificando os dados
df.info()
# Olhando os dados
df.head()
df.isnull().sum()
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
# Distribuição dos dados sobre o valor do aluguel
df['rent amount (R$)'].value_counts()
# Distribuição dos dados sobre o valor da taxa do imóvel
df['property tax (R$)'].value_counts()
# Dsitribuição dos dados sobre o valor do seguro incendio
df['fire insurance (R$)'].value_counts()
# Verificando o imovel com 13 quartos
df[df['rooms'] == 13]
# Olhando a correlação das variáveis
import seaborn as sns

sns.heatmap(df.corr(), annot=True, linewidths=0.5)
# Tratamento de dados

# Convertendo a coluna floor
df[df['floor'] == '-'] = '0'
df['floor'] = df['floor'].astype(int)

# Convertendo as colunas categóricas
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes
df.info()
# Distribuição de todas as variáveis
df.describe()
from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
df['animal'] = la.fit_transform(df['animal'])
df['furniture'] =la.fit_transform(df['furniture'])
df['city'] =la.fit_transform(df['city'])
df['animal'].value_counts()
df['city'].value_counts()
# Importando as bibliotecas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
# Divisão dos dados
train, test = train_test_split(df)

train.shape, test.shape
feats = [col for col in df.columns if col not in ['total (R$)']]

dt = DecisionTreeRegressor()
dt.fit(train[feats], train['total (R$)'])
preds0 = dt.predict(test[feats])
print(mean_squared_error(test['total (R$)'], preds0))
from xgboost import XGBRegressor
feats = [col for col in df.columns if col not in ['total (R$)']]

xgb = XGBRegressor()
xgb.fit(train[feats], train['total (R$)'])
preds3 = xgb.predict(test[feats])
print(mean_squared_error(test['total (R$)'], preds3))
feats = [col for col in df.columns if col not in ['total (R$)']]

gbr = GradientBoostingRegressor()
gbr.fit(train[feats], train['total (R$)'])
preds4 = dt.predict(test[feats])
print(mean_squared_error(test['total (R$)'], preds4))
from sklearn import linear_model
feats = [col for col in df.columns if col not in ['total (R$)']]

lm = linear_model.LinearRegression()
lm.fit(train[feats], train['total (R$)'])
preds5 = lm.predict(test[feats])
print(mean_squared_error(test['total (R$)'], preds5))
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
print(mean_squared_error(test['total (R$)'], preds1))
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
print(mean_squared_error(test['total (R$)'], preds2))
# Atividade para 17/09/2020

# - Melhorar tratamento dos dados
# - Possibilidade de feature engineering
# - Utilização de outros algoritimos
# - OBJETIVO: Melhorar o desempenho da previsão