import numpy as np

import pandas as pd

from pandas_datareader import data as wb
indices = ['^bvsp','wege3.sa','shul4.sa']

b3 = pd.DataFrame()
for a in indices: 

    b3[a] = wb.DataReader(a, data_source ='yahoo', start = '2010-01-01', end='2020-09-21')['Close']
b3
b3.columns = ['IBOV','WEGE3', 'SHUL4']

b3
weg = b3.drop(['SHUL4'], axis = 1)

schulz = b3.drop(['WEGE3'], axis = 1)
weg
schulz
b3_ = (b3 / b3.iloc[0] * 100)
b3_
import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

plt.style.use('classic')

import seaborn as sns

%matplotlib inline
weg2 = weg.loc['2017-01-01' :]
sns.set()

weg2['WEGE3'].plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('Cotações da WEG últimos 3 anos', fontsize = 20);

plt.show()
sns.set()

weg2 = weg.loc['2019-08-01' :]

weg2['WEGE3'].plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('Cotações da WEG últimos 12 meses', fontsize = 20);

plt.show()
sns.set()

weg2 = weg.loc['2020-03-01' :]

weg2['WEGE3'].plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('Cotações da WEG últimos 6 meses', fontsize = 20);

plt.show()
sns.set()

weg2 = weg.loc['2020-06-01' :]

weg2['WEGE3'].plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('Cotações da WEG últimos 3 meses', fontsize = 20);

plt.show()
sns.set()

b3_2 = b3.drop(['IBOV'], axis = 1)

b3_2 = b3_2.loc['2015-01-01' :]

b3_2.plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('WEG x SCHULZ', fontsize = 20);

plt.show()
b3_.plot(figsize=(20, 8))

plt.ylabel('')

plt.xlabel('')

plt.legend(fancybox=True, framealpha=1 , shadow=True, borderpad=0.5, loc='upper left')

plt.title('WEG, SCHULZ & IBOV', fontsize = 20);

plt.show()