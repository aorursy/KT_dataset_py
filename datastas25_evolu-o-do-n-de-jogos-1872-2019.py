import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
result=pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv' )
result.tail()
result['date']=pd.to_datetime(result['date']).dt.year
result['date']
result=result[result['date']<2020]
jogos=result.groupby('date').agg({'neutral':'count'})
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(jogos)

ax.set_title('Evolução do Nº de Jogos pelas Seleções')

ax.set_xlabel('Ano')

ax.set_ylabel('Jogos')