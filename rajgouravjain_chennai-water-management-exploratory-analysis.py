# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.plotly as py



import cufflinks as cf

%matplotlib inline



init_notebook_mode(connected=True)

cf.go_offline()



plt.style.use('seaborn')
df_rlevels = pd.read_csv('../input/chennai_reservoir_levels.csv')

df_rlevels.head()



df_rlevels.describe()
df_rlevels['Month'] = df_rlevels['Date'].apply(lambda s: s.split('-')[1])
df_rlevels['Year'] = df_rlevels['Date'].apply(lambda s: s.split('-')[2])
df_rlevels['Date'] = pd.to_datetime(df_rlevels['Date'], format='%d-%m-%Y')

df_rlevels.set_index('Date',inplace=True)



df_rlevels['Total'] = df_rlevels['POONDI'] + df_rlevels['CHOLAVARAM'] + df_rlevels['REDHILLS'] +df_rlevels['CHEMBARAMBAKKAM']
df_rlevels.head()
df_rlevels[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']].iplot()
corr_rlevels = df_rlevels.corr()

corr_rlevels
df_rlevels[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']].iplot(kind='box')
df_rlevels[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']].iplot(kind='hist',bins=100)
df_rfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv')
df_rfall.info()
df_rfall.head()
df_rfall['Month'] = df_rfall['Date'].apply(lambda s: s.split('-')[1])

df_rfall['Year'] = df_rfall['Date'].apply(lambda s: s.split('-')[2])

df_rfall['Date'] = pd.to_datetime(df_rfall['Date'], format='%d-%m-%Y')

df_rfall.set_index('Date',inplace=True)
df_rfall[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']].iplot()
corr_rfall = df_rfall.corr()

corr_rfall
df_rfall['Total'] = df_rfall['POONDI'] + df_rfall['CHOLAVARAM'] + df_rfall['REDHILLS'] +df_rfall['CHEMBARAMBAKKAM']

sns.boxplot(x='Month', y='Total',data=df_rfall)
sns.boxplot(x='Year', y='Total',data=df_rfall)
sns.boxplot(x='Month', y='Total',data=df_rlevels)

#df_rlevels.iplot(kind='box',x='Month',y='Total')
sns.boxplot(x='Year', y='Total',data=df_rlevels)

df_rfall
df_poondi =  pd.concat([df_rlevels['POONDI'], df_rfall['POONDI'],df_rlevels['Year'],df_rlevels['Month']], axis=1, keys=['Poondi_level', 'Poondi_rainfall', 'Year','Month'])
df_poondi['Poondi_level'] = df_poondi['Poondi_level'].apply(float)

df_poondi['Poondi_rainfall'] = df_poondi['Poondi_rainfall'].apply(float)
df_poondi[ (df_poondi.index > pd.to_datetime('2010/11/12')) &  (df_poondi.index < pd.to_datetime('2010/11/18'))]
df_poondi.iplot(kind='bar',x='Month',y='Poondi_rainfall')
df_poondi.iplot(kind='bar',x='Month',y='Poondi_level')
sns.scatterplot(x='Poondi_rainfall',y='Poondi_level',data=df_poondi)