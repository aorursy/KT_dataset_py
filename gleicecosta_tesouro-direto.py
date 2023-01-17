# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_op = pd.read_csv ('../input/OperacoesTesouroDireto.csv', sep='[;]', engine='python')

df_ve = pd.read_csv ('../input/VendasTesouroDireto.csv', sep='[;]', engine='python')

df_es = pd.read_csv ('../input/EstoqueTesouroDireto.csv', sep='[;]', engine='python')
df_op.shape, df_ve.shape, df_es.shape
df_op.head().T
df_ve.head(5).T
df_es.head(5).T
df_op['Tipo da Operacao'].value_counts().plot.bar(figsize=(8,6), title=('Tipo da Operação'))
df_op.groupby('Tipo da Operacao')[u'Tipo Titulo'].value_counts()
df_op.groupby('Tipo da Operacao')[u'Data da Operacao'].value_counts()
df_op.info()
df_op.isnull().sum()
df_op['Valor do Titulo'].isnull().sum()
df_op.tail(10).T
#df_op.dropna(inplace=True)
df_op.tail(1).T
df_op.isnull().sum()
df_op.head().T
#df_op['Canal da Operacao'] = df_op['Canal da Operacao'].astype('category').cat.codes

#df_op['Tipo da Operacao'] = df_op['Tipo da Operacao'].astype('category').cat.codes
#df_op['Vencimento do Titulo'] = pd.to_datetime(df_op['Vencimento do Titulo'])

#df_op['Data da Operacao'] = pd.to_datetime(df_op['Data da Operacao'])
df_op.head().T
df_op.info()
df_op['Tipo Titulo'].value_counts()
df_op['Tipo Titulo'].value_counts().plot.barh(figsize=(12,6), title=('Tipo do Titulo'))
df_ve.groupby('Vencimento do Titulo')[u'Data Venda'].value_counts()
df_ve.info()
#df_ve['Vencimento do Titulo'] = pd.to_datetime(df_ve['Vencimento do Titulo'])

#df_ve['Data Venda'] = pd.to_datetime(df_ve['Data Venda'])
df_ve['Tipo Titulo'].value_counts().plot.barh(figsize=(10,4), title=('Tipo do Titulo Vendidos'))
#df_es['Vencimento do Titulo'] = pd.to_datetime(df_es['Vencimento do Titulo'])
df_es.info()
#https://stackoverflow.com/questions/38333954/converting-object-to-datetime-format-in-python

#https://www.alura.com.br/artigos/lidando-com-datas-e-horarios-no-python

#https://pythonhelp.wordpress.com/2012/07/10/trabalhando-com-datas-e-horas-em-python-datetime/

#https://dicasdepython.com.br/python-como-converter-string-em-date/
df_op.info()
df_ve.info()
df_es.info()
df_es['Valor Estoque'].value_counts().plot.barh(figsize=(30,20), title=('Valor do Estoque'))
df_es.groupby('Tipo Titulo')['Quantidade'].value_counts().plot.barh(figsize=(30,20), title=('Tipo do Titulo'))
df_op.corr()
import seaborn as sns

sns.heatmap(df_op.corr())
df_ve.corr()
sns.heatmap(df_ve.corr())
df_es.corr()
sns.heatmap(df_es.corr())