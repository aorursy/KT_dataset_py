import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime
parser = (lambda x:datetime.datetime.strptime(x, '%Y.%m.%d')) 

df = pd.read_csv('../input/sp-beaches-water-quality/sp_beaches.csv', parse_dates=['Date'])

df.head()
print(f'Numero de praias: {len(df.Beach.unique())}') 

print(f'Numero de cidades: {len(df.City.unique())}')

print('Dados de: {} até {}'.format(df.Date.min().year, df.Date.max().year ))

print(df.isnull().sum(axis=0)) 

df.info() # Nao tem dados faltando