# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carregando os dados

df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
# Juntando os dataframes

df_all = df.append(test)



df_all.shape
# Verificando tamanhos e tipos

df_all.info()
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Olhando a coluna dependency

df_all['dependency'].value_counts()
# Analisando os dados da coluna edjefa

df_all['edjefa'].value_counts()
# Analisando os dados da coluna edjefe

df_all['edjefe'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0

# nas colunas edjefa e edjefe

mapeamento = {'yes': 1, 'no': 0}



df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Olhando a coluna dependency

df_all['dependency'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0

# na coluna dependency

df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Visualizando do comando info

df_all.info()
# Verificando os valores nulos

df_all.isnull().sum()
 # Verificando os valores de aluguel (v2a1) para os chefes/as de familia (parentesco1 = 1)

df_all[df_all['parentesco1'] == 1]['v2a1'].isnull().sum()
# Qual a cara dos dados de v18q

df_all['v18q'].value_counts()
# Prenchendo com -1 os valores nulos de v2a1

df_all['v2a1'].fillna(-1, inplace=True)
# Prenchendo com 0 os valores nulos de v18q1

df_all['v18q1'].fillna(0, inplace=True)
# Verificando os valores nulos

df_all.isnull().sum().sort_values()
# Prenchendo com -1 os valores nulos de SQBmeaned, meaneduc e rez_esc

df_all['SQBmeaned'].fillna(-1, inplace=True)

df_all['meaneduc'].fillna(-1, inplace=True)

df_all['rez_esc'].fillna(-1, inplace=True)
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
# Separar os dataframes

train, test = df_all[~df_all['Target'].isnull()], df_all[df_all['Target'].isnull()]



train.shape, test.shape
# Instanciando o random forest classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt



fig=plt.figure(figsize=(15, 20))



# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Verificando a classe target nos dados de treino

train['Target'].value_counts(normalize=True)
# Limitando o treinamento as/aos chefas/es de familia



# Coluna parentesco1

heads = train[train['parentesco1'] == 1]
# Criando, treinando, fazendo previsões e gerando o arquivo de submissão com RF2

# Dados de treinao apenas dos chefes/chefas de familia



rf2 = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)



rf2.fit(heads[feats], heads['Target'])



test['Target'] = rf2.predict(test[feats]).astype(int)



test[['Id', 'Target']].to_csv('submission.csv', index=False)
# Qual o tamanho da base de treino heads?

heads.shape
# Verificando os valores da coluna hhsize

train['hhsize'].value_counts()
# Verificando os dados da coluna tamviv

train['tamviv'].value_counts()
# Verificando os dados da coluna tamhog

train['tamhog'].value_counts()
# Feature Engineering / Criação de novas colunas



# Relação tamanho da casa / moradores

df_all['hhsize-pc'] = df_all['hhsize'] / df_all['tamviv']



# Relação qtde celulares / moradores

df_all['mobile-pc'] = df_all['qmobilephone'] / df_all['tamviv']



# Relaçao qtde de tablets / moradores

df_all['tablet-pc'] = df_all['v18q1'] / df_all['tamviv']



# Relação qtde de quartos / moradores

df_all['rooms-pc'] = df_all['rooms'] / df_all['tamviv']
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
# Separar os dataframes

train, test = df_all[~df_all['Target'].isnull()], df_all[df_all['Target'].isnull()]



train.shape, test.shape
# Criando, treinando, fazendo previsões e gerando o arquivo de submissão com RF3

# Dados de treino com 4 colunas a mais



rf3 = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)



rf3.fit(train[feats], train['Target'])



test['Target'] = rf3.predict(test[feats]).astype(int)



test[['Id', 'Target']].to_csv('submission.csv', index=False)
fig=plt.figure(figsize=(15, 20))



# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf3.feature_importances_, index=feats).sort_values().plot.barh()
# Juntando as abordagens



# Selecionando para treio só parentesco1 == 1

heads2 = train[train['parentesco1'] == 1]
# Criando, treinando, fazendo previsões e gerando o arquivo de submissão com RF4

# Dados de treino apenas dos chefes/chefas de familia e 4 colunas a mais



rf4 = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)



rf4.fit(heads2[feats], heads2['Target'])



test['Target'] = rf4.predict(test[feats]).astype(int)



test[['Id', 'Target']].to_csv('submission.csv', index=False)