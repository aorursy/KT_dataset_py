# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import requests

import json

from pandas.io.json import json_normalize



import matplotlib.pyplot as plt

import seaborn as sns



sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Consumindo dados Via API

    #cUrl = "https://olinda.bcb.gov.br/olinda/servico/Informes_Correspondentes/versao/v1/odata/Correspondentes?$top=100&$format=json&$select=CnpjContratante,NomeContratante,CnpjCorrespondente,NomeCorrespondente,Tipo,Ordem,MunicipioIBGE,Municipio,UF,ServicosCorrespondentes,Posicao"

    #cUrl = "https://olinda.bcb.gov.br/olinda/servico/Informes_Correspondentes/versao/v1/odata/Correspondentes?$top=10000&$format=json"

    #cUrl = "https://olinda.bcb.gov.br/olinda/servico/Informes_Correspondentes/versao/v1/odata/Correspondentes?$format=json"

    #dados = pd.read_json(cUrl)

#Acessando Json baixado no dia 03/11/2019 utilizando a url :

    #https://olinda.bcb.gov.br/olinda/servico/Informes_Correspondentes/versao/v1/odata/Correspondentes?$top=1000000000000&$format=json     



dados = pd.read_json("/kaggle/input/financial-institutions-by-cities-in-brazil/Correspondentes.json")    

df = pd.DataFrame(dados) 

df.head()
dfnew = (json_normalize(df.value))

dfnew.head()
dfnew['urlserv'] = df['@odata.context'] 

dfnew.head()
dfnew.shape
dfnew.columns
dfnew.info()
df1 = dfnew.dropna()

df2 = dfnew[pd.isnull(dfnew).any(axis=1)]
df1.head()
df2.head()
df1.shape
df1.nunique()
df1["Tipo"].value_counts()
fig, ax = plt.subplots()

fig.set_size_inches(8, 9)

ax = sns.countplot(x="Tipo", data=df1, palette="Accent")
df1["UF"].value_counts(normalize=True)
#Distribuição de Correspondentes por UF

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax = sns.countplot(y=df1['UF'], data=df1) 
df1["Municipio"].value_counts().head(50)
df1["Municipio"].value_counts().head(10).plot.barh()
df1["NomeContratante"].value_counts().head(50)