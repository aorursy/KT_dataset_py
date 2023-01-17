# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
temp = pd.read_csv('../input/metereologia-cuiaba-cleaning/Meteo_Cuiaba_Temperaturas_Ready.csv')

temp.head()
temp.info()
temp = temp.drop(['Unnamed: 0'],axis='columns')
temp['Data'] = pd.to_datetime(temp['Data'],yearfirst=True)
temp['mes'] = temp.apply(lambda x: x['Data'].month,axis=1)
temp['ano'] = temp.apply(lambda x: x['Data'].year,axis=1)
num_dias = temp.count()[0]
temp['Data'].describe()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')
print('mean: ',temp['temp_max'].mean())

print('std: ',temp['temp_max'].std())

ax = sns.distplot(temp['temp_max'],bins=100,norm_hist=True,color='red')

ax.set(xlabel='$^{o}C$',ylabel='%',title='Histograma - Máxima Diária')

plt.savefig('Histograma.png',dpi=300)
d36 = temp[temp['temp_max']>=36]['temp_max'].count()

d40 = temp[temp['temp_max']>=40]['temp_max'].count()

d43 = temp[temp['temp_max']>=43]['temp_max'].count()
print([d36,d40,d43])

print([d36,d40,d43]/num_dias*100)
q1 = temp['temp_max'] >= 36

q2 = temp['temp_max']<=40

print(temp.loc[q1 & q2].count()[0],' dias')

print(temp.loc[q1 & q2].count()[0]/num_dias*100,'%')
print(temp['temp_max'].mode()[0])
q1 = temp['temp_min'] == temp['temp_min'].min()

q2 = temp['temp_max'] == temp['temp_max'].max()
temp.loc[q1][['Data','temp_min']]
temp.loc[q2][['Data','temp_max']]
group = temp.groupby('mes')
mes_quente = pd.DataFrame(group.sum()['temp_med'].sort_values(ascending=False).head(3).reset_index())
mes_quente.head(3)
ax = sns.barplot(x='mes',y='temp_med',data=mes_quente,order=mes_quente['mes'].values,color='#11688d')

ax.set(xlabel='Mês',ylabel='Temperatura Média Acumulada',title='Mês mais quente',ylim=(43000,46000))

plt.savefig('Mes_mais_calor.png',dpi=300)
sns.lineplot(x='mes',y='temp_max',data=temp,)
sns.scatterplot(x='Data',y='temp_med',data=temp,hue='mes')
anual = temp.groupby('ano')
max_temp_anual = pd.DataFrame(anual.max()['temp_max']).reset_index()

min_temp_anual = pd.DataFrame(anual.max()['temp_min']).reset_index()
ax = sns.lmplot(x='ano',y='temp_max',data=max_temp_anual)

ax.set(xlabel='Ano',ylabel='$^{o}C$',title='Regressão Linear')

plt.savefig('Regressao.png',dpi=300)
ax = sns.lineplot(x='ano',y='temp_max',data=max_temp_anual)

ax.set(xlabel='Ano',ylabel='$^{o}C$',title='Temperatura Máxima Anual')

plt.savefig('Maxima_anual.png',dpi=300)
years = []



maximo = -100

for i in range(len(max_temp_anual)):



    current = max_temp_anual['temp_max'].loc[i]

    

    if current > maximo:

        maximo = current

        years.append([max_temp_anual['ano'].loc[i],current])
intervalo = pd.DataFrame(years[1:],columns=['ano','temp_max'])
intervalo['delta'] = intervalo['ano'].diff()
intervalo
sns.lineplot(x='ano',y='delta',data=intervalo)