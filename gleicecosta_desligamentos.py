# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy     as np # linear algebra

import pandas    as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime  as dt # Funções de data e hora

import seaborn   as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_excel('../input/Apurao mensal de indicadores_BD Deslig e Pert_janela OUT17 a SET18.xlsx', sheet_name='Desligamentos', header=0)

df2 = pd.read_excel('../input/Apurao mensal de indicadores_BD Deslig e Pert_janela NOV17 a OUT18.xlsx', sheet_name='Desligamentos', header=0)
df1.rename(columns={'Unnamed: 3':'DataEntregaAge'}, inplace=True)

df1.rename(columns={'Unnamed: 4':'DataReligEfetivo'}, inplace=True)
df1.tail()
df2.sample(5)
df1 = df1[df1.DataDeslForc<dt.datetime(2017,11,1)]
df3 = df1

df3 = df3.append(df2)
df1.count()
df1.columns
df2.count()
df2.columns
df3.count()
df3.head().T
df3['DataDeslForc'].sample(10)
type(df3['DataDeslForc'])
df3.index
df3.isnull().sum()
df3['UF'] = df3['EqpId'].str[0:2]
df3.head().T
%matplotlib inline
df4 = df3

df4 = df4[['CodOrigem', 'DescOrigem']]

df4 = df4.drop_duplicates('CodOrigem')

df4.to_dict
explode = (0.4, 0.4, 0.4, 0, 0 )

df3['DescOrigem'].value_counts().sort_values().plot.pie(explode =explode, autopct='%1.1f%%', shadow=True, startangle=140)
df3 = df3[(df3['CodOrigem'] == 'I') | (df3['CodOrigem'] == 'S')]
df3['CodLocal'].value_counts().plot.bar(figsize=(14,10))
df5 = df3

df5 = df5[['CodLocal', 'DescLocal']]

df5 = df5.drop_duplicates('CodLocal')

df5

# df5.to_dict
df3['DescCausa'].value_counts().plot.barh(figsize=(14,30), title=('Causas dos Desligamentos'))
df6 = df3

df6 = df6[['CodCausa', 'DescCausa']]

df6 = df6.drop_duplicates('CodCausa')

df6

df6.to_dict
df3['UF'].value_counts().sort_values().plot.barh(figsize=(12,8), title=('Quantidade de Desligamentos por UF'))
#df3['ano']=dt.datetime(df3['DataDeslForc'].year,1,1)

from datetime import date

#df3['DataDeslForc'].dt.strftime('%Y')

#df3['DataDeslForc'].dt.strftime('%m')

df3['DataDeslForc'].dt.year

df3['DataDeslForc'].dt.month

#df3['AnoDesl'] = df3['DataDeslForc'].dt.strftime('%Y')

#df3['MesDesl'] = df3['DataDeslForc'].dt.strftime('%m')

df3['AnoDesl'] = df3['DataDeslForc'].dt.year

df3['MesDesl'] = df3['DataDeslForc'].dt.month

#hj = date(df3['DataDeslForc']).year

df3.sample(15).T
df3['AnoDesl'].value_counts().sort_values().plot.pie(title=('Percentual de Desligamentos por Ano'), figsize=(8,8),autopct='%1.1f%%', startangle=140)
df3['MesDesl'].value_counts().sort_values().plot.bar(figsize=(4,5), title=('Quantidade de Desligamentos por mês'))