import pandas as pd

import numpy as np

import matplotlib

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
arquivo = '/kaggle/input/fifa19eda/fifa_eda.csv'

dataset = pd.read_csv(arquivo, sep=',' ,header=0)

dataset.shape
type(dataset)
dataset.head()
dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset.dropna(how='all', inplace=True)
#preenche com -1 os values null da coluna 

dataset['International Reputation'].fillna(-1, inplace=True)

dataset['Skill Moves'].fillna(-1, inplace=True)

dataset['Club'].fillna(0, inplace=True)

dataset['Value'].fillna(0, inplace=True)

dataset['Contract Valid Until'].fillna(0, inplace=True)
dataset.describe()


pe = dataset['Preferred Foot'].value_counts()

plot = pe.plot.bar()
dataset.loc[dataset['Nationality']=='Brazil']
#Quantidade de brasileiros

dataset.loc[dataset['Nationality']=='Brazil'].count()
dataset.loc[dataset['International Reputation']== 5] 



salario = dataset.loc[dataset.Nationality=='Brazil','Wage']

salario.plot.hist()


import matplotlib

%matplotlib inline

idades =  dataset['Age'].value_counts()[:10]



plot = idades.plot.bar()


pais = dataset['Nationality'].value_counts()[:10]



plot = pais.plot.bar()

potencial = dataset.loc[dataset['Potential']> 90] 

potencial['Name']

br = dataset[(dataset.Nationality == 'Brazil') & (dataset.Potential >= 90)]

br['Name']
%matplotlib inline

dataset.plot(x='Wage',y='Potential',kind='scatter', title='Salário x Potencial',color='r')
dataset.describe()
dataset.plot(x='Potential',y='Skill Moves',kind='scatter', title='Potential x Skill Moves',color='r')
#media das salarios

np.mean(dataset['Wage'])
#desvio padrão das idades

np.std(dataset['Age'])
#media de brasileiros

np.mean(dataset['Nationality']=='Brazil')
dataset.plot(x='Wage',y='Release Clause',kind='scatter', title='Wage x Release Clause',color='c')
dataset.plot(x='Joined',y='Release Clause',kind='scatter', title='Joined x Release CLause',color='c')
dataset.plot(x='Joined',y='Wage',kind='scatter', title='Joined x Wage',color='c')
dataset.columns
X_data = dataset[['Age','Overall', 'Potential',

       'Value', 'Wage','International Reputation',

       'Skill Moves','Height','Weight', 'Release Clause']]

Y_data = dataset[['ID', 'Name','Nationality','Joined']]
X_data = X_data.apply(lambda x: (x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
X_data
X_data.describe()