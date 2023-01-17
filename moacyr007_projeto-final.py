import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly

plotly.offline.init_notebook_mode()

import datetime

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        
dataset=pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv',sep='\t')

dataset['DATA INICIAL']=pd.to_datetime(dataset['DATA INICIAL'])

dataset['DATA FINAL']=pd.to_datetime(dataset['DATA FINAL'])
EtanolData=dataset.loc[dataset['PRODUTO'] == 'ETANOL HIDRATADO']

GasolinaData=dataset.loc[dataset['PRODUTO'] == 'GASOLINA COMUM']

DieselData=dataset.loc[dataset['PRODUTO'] == 'ÓLEO DIESEL']

Diesel10Data=dataset.loc[dataset['PRODUTO'] == 'ÓLEO DIESEL S10']

GLPData=dataset.loc[dataset['PRODUTO'] == 'GLP']

GNVData=dataset.loc[dataset['PRODUTO'] == 'GNV']

x1=GasolinaData['ANO'].loc[dataset['REGIÃO']=='SUDESTE']

y1=GasolinaData['PREÇO MÉDIO REVENDA'].loc[dataset['REGIÃO']=='SUDESTE']

data=[go.Scatter(x=x1,y=y1)]

fig=go.Figure(data=data)

py.iplot(fig)
x1=np.array(GasolinaData['ANO'].loc[dataset['REGIÃO']=='SUDESTE']).reshape(-1, 1)

y1=GasolinaData['PREÇO MÉDIO REVENDA'].loc[dataset['REGIÃO']=='SUDESTE']

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2) 
linreg = LinearRegression()

linreg.fit(x_train, y_train)
print(linreg.intercept_)

print(linreg.coef_)
linreg.predict(np.array(2025).reshape(-1,1))
#Qual a região com o combustível x mais caro?

#Modelo 2



dataset=pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv',sep='\t')

dataset['DATA INICIAL']=pd.to_datetime(dataset['DATA INICIAL'])

dataset['DATA FINAL']=pd.to_datetime(dataset['DATA FINAL'])



print(dataset)