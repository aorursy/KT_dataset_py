#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#importando os pacotes

df_train = pd.read_csv('../input/train.csv')

#visualizando colunas

df_train.columns
#Descrição estatisticas 

df_train['GarageFinish'].describe()
#histograma

sns.distplot(df_train['SalePrice']);

#fig.axis(ymin=0, ymax=800000);#fixa um valor máximo a escal Y

#plt.xticks(rotation=45);# altera a rotação da escala X



#box plot GarageCars/saleprice

var = 'GarageCars'# declarando a variavel que aparecerá na escala X

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data);

fig.axis(ymin=0, ymax=800000);#fixa um valor máximo a escal Y

plt.xticks(rotation=50);# altera a rotação da escala X
#explorando os diagramas de acordo com a variávis

sns.set()

cols = ['OverallCond','SalePrice','OverallQual']

sns.pairplot(df_train[cols], size = 2.5)

fig.axis(ymin=0, ymax=800000);#fixa um valor máximo a escal Y

plt.xticks(rotation=10);# altera a rotação da escala X

plt.show();
#histogram and normal probability plot

#sns.distplot(df_train['SalePrice'], fit=norm);

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#Dados ausentes

total = df_train.isnull().sum().sort_values(ascending=False)

missing_data = pd.concat([total], axis=1, keys= ['Total'])

missing_data.head(20) #leitura da quantidade de linhas
#Dados ausentes,podem implicar no dados do tamanho da amostra, importande identificarmos e eliminarmos para 

#obtermos um padrão.



total = df_train.isnull().sum().sort_values(ascending= False)

porcentagem = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, porcentagem], axis=1, keys=['Total', 'Percent %'])

missing_data.head(20) #leitura da quantidade de linhas

#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#Dados Ausentes

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#deletando dados ausentes

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #verificação se não ha dados
#padronizando os dados

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#analise binária saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deletando pontos dispersos (out liars)

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

print(df_train)
#analise binária pos deleção saleprice/TotalBsmtSF

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histograma e normalidade normal

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#transforma de log

df_train['SalePrice'] = np.log(df_train['SalePrice'])

print(df_train)
#histograma transformado e probabilidade Normal plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)

print(df_train)