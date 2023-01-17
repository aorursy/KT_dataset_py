# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy     as np # linear algebra

import pandas    as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime  as dt # Funções de data e hora



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_excel('../input/Apurao mensal de indicadores_BD Deslig e Pert_janela OUT17 a SET18.xlsx', sheet_name='Desligamentos', header=0)

df2 = pd.read_excel('../input/Apurao mensal de indicadores_BD Deslig e Pert_janela NOV17 a OUT18.xlsx', sheet_name='Desligamentos', header=0)
df1.rename(columns={'Unnamed: 3':'DataEntregaAge'}, inplace=True)

df1.rename(columns={'Unnamed: 4':'DataReligEfetivo'}, inplace=True)
df1.tail(15)
df2.sample(5)
df1 = df1[df1.DataDeslForc<dt.datetime(2017,11,1)]
df3 = df1

df3 = df3.append(df2)
df1.count()
df1.columns
df3[df3['CodOrigem'] == 'I'].T
df3[df3['CodOrigem'] != 'E'].T
df3[df3['CodOrigem'] != 'O'].T
df3[df3['CodOrigem'].isin (['I' , 'S'])].T
df4 = df3

df4 = df4[['CodOrigem', 'DescOrigem']]

df4 = df4.drop_duplicates('CodOrigem')

df4

# df4.to_dict
df3 = df3[(df3['CodOrigem'] == 'I') | (df3['CodOrigem'] == 'S')]
df3['UF'] = df3['EqpId'].str[0:2]
df1.columns
df2.count()
df3.count()
df3.head().T
df3['DataDeslForc'].sample(10)
type(df3['DataDeslForc'])
df3.index
%matplotlib inline
df3['UF'].value_counts().plot.bar()
df3['UF'].value_counts().sort_values().plot.barh()
df3.isnull().sum()