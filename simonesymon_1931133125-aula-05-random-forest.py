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
df.head()
df.Target.count_values
# Quais colunas do dataframe são do tipo object
df_all.select_dtypes('object').head()
# Analisando os dados da coluna edjefa
df_all['edjefa'].value_counts()
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
# Verificando os valores nulos
df_all.isnull().sum()
# Prrenchendo com -1 todos os valores nulos
df_all.fillna(-1, inplace=True)
# Separando as colunas para treinamento
feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
# Separar os dataframes
train, test = df_all[df_all['Target'] != -1], df_all[df_all['Target'] == -1]
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