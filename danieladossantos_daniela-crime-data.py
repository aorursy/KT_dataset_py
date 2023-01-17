# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/crime-data-in-brazil/BO_2014_1.csv')
df.head()
df['RUBRICA'].unique() 
df1 = df.copy()
df1[['RUBRICA']] = df1['RUBRICA'].apply(lambda x: ''.join(['(', x.split('(')[1]]).strip())
temp = df1[['RUBRICA', 'ID_DELEGACIA']].groupby('RUBRICA', as_index=False).count().sort_values(by='ID_DELEGACIA', ascending=False)[:10]

fig = go.Figure()
fig.add_trace(go.Bar(x=temp['RUBRICA'], y=temp['ID_DELEGACIA']))
fig.update_layout(title='OS 10 CRIMES MAIS PRATICADOS EM SÃO PAULO EM 2014.1')
fig.show()
cs=['SEXO_PESSOA', 'IDADE_PESSOA']
df2=df.filter(items=cs) 
df2.head()
df2.groupby(['SEXO_PESSOA']).mean().drop(['I'],axis=0)
colors = ['Orchid', 'RoyalBlue']

temp = df[['SEXO_PESSOA', 'IDADE_PESSOA']].loc[
    (
        (
            (df['SEXO_PESSOA'].str.contains('F')) | 
            (df['SEXO_PESSOA'].str.contains('M'))
            
        )
        & (df['SEXO_PESSOA'].isna() == False)
    )
].groupby(['SEXO_PESSOA'], as_index=False).mean()

temp.loc[temp['SEXO_PESSOA'] == 'M', 'SEXO_PESSOA'] = 'Masculino'
temp.loc[temp['SEXO_PESSOA'] == 'F', 'SEXO_PESSOA'] = 'Feminino'

fig = go.Figure()
fig.add_trace(go.Pie(labels=temp['SEXO_PESSOA'], values=temp['IDADE_PESSOA'], textinfo='percent',marker=dict(colors=colors)) ) # textinfo determina qual a informação será exibida, percent para porcentagem
fig.update_layout(title='MÉDIA DE IDADE DOS AUTORES DOS CRIMES POR GÊNERO')
fig.show()
temp = df[['SEXO_PESSOA', 'DESCR_TIPO_PESSOA']].loc[
     (
        (
            (df['SEXO_PESSOA'].str.contains('F')) | 
            (df['SEXO_PESSOA'].str.contains('M'))
            
        )
        & (df['SEXO_PESSOA'].isna() == False)
    )&(
        
        (
            (df['DESCR_TIPO_PESSOA'].str.contains('Vitima')) | 
            (df['DESCR_TIPO_PESSOA'].str.contains('Vítima'))
        )
        
     & (df['DESCR_TIPO_PESSOA'].isna() == False)
        
    )
].groupby(['SEXO_PESSOA'], as_index=False).count()

temp.loc[temp['SEXO_PESSOA'] == 'M', 'SEXO_PESSOA'] = 'Masculino'
temp.loc[temp['SEXO_PESSOA'] == 'F', 'SEXO_PESSOA'] = 'Feminino'

fig = go.Figure()
fig.add_trace(go.Pie(labels=temp['SEXO_PESSOA'], values=temp['DESCR_TIPO_PESSOA'],marker=dict(colors=colors)))
fig.update_layout(title='QUANTIDADE DE VÍTIMAS POR GÊNERO')
fig.show()
temp = df[['RUBRICA', 'CIDADE']].groupby('CIDADE', as_index=False).count().sort_values(by='RUBRICA', ascending=False)[:5]
temp.head()
fig = go.Figure()
fig.add_trace(go.Bar(x=temp['CIDADE'], y=temp['RUBRICA'],text=temp['RUBRICA'],textposition='auto'))
fig.update_layout(title='5 CIDADES DE SÃO PAULO COM MAIOR REGISTRO DE CRIMES EM 2014.1')
fig.show()
