# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_houses = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df_houses.head(5)
#Renomeando a coluna de valor do aluguel
df_houses.rename(columns={'rent amount (R$)' : 'valor_aluguel'},inplace = True)
df_houses['valor_aluguel'].mean()
df_houses['valor_aluguel'].median()
df_houses['valor_aluguel'].std()
df_houses['valor_aluguel'].describe()
plt.figure(figsize=(12,6))
df_houses['valor_aluguel'].plot(kind='hist',bins=100);
# análise skewness
df_houses['valor_aluguel'].skew()
# Kurtosis - Curva Leptocurtica (kurtosis > 0)
df_houses['valor_aluguel'].kurtosis()
#Qual a cidade com a média de aluguel mais alta?
df_houses.groupby('city')['valor_aluguel'].mean().reset_index().sort_values('valor_aluguel',ascending=False)
#Criando uma nova coluna para classificar alugueis como valores altos e baixos
df_houses['aluguel_alto'] = ['Alto' if x > 5000 else 'Baixo' for x in df_houses['valor_aluguel']]
#Média do número de banheiros por faixa de aluguel
df_houses.groupby('aluguel_alto')['bathroom'].mean()
#Análise da quantidade de banheiros por regiões em aluguéis considerados altos
df_houses.groupby(['aluguel_alto','city'])['bathroom'].mean()['Alto']
#Filtrando o dataset e fazendo o cálculo da porcentagem
df_houses.query('aluguel_alto == "Alto" & animal == "acept"').shape[0] / df_houses.query('aluguel_alto == "Alto"').shape[0]
#Avaliando o total de imóveis de alto valor que aceitam animais por cidade
df_houses.query('aluguel_alto == "Alto" & animal == "acept"').groupby('city').count()['animal']
#Calculo da porcentagem de aceitação por região.
df_houses.query('aluguel_alto == "Alto" & animal == "acept"').groupby('city').count()['animal'] / df_houses.query('aluguel_alto == "Alto"').groupby('city').count()['animal']
#Cálculo da %
df_houses.query('aluguel_alto == "Alto" & furniture == "furnished"').shape[0] / df_houses.query('aluguel_alto == "Alto"').shape[0]

#Calculo da porcentagem por região.
df_houses.query('aluguel_alto == "Alto" & furniture == "furnished"').groupby('city').count()['furniture'] / df_houses.query('aluguel_alto == "Alto"').groupby('city').count()['furniture']
plt.figure(figsize=(12,6))
sns.heatmap(df_houses.corr(method='spearman'), annot=True);
