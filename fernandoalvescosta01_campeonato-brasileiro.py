# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns



# Any results you write to the current directory are saved as output.
dados = pd.read_csv('/kaggle/input/campeonato-braileiro-20092018/tabelas/Tabela_Clubes.csv')
dados
dados = dados.drop('Unnamed: 13', 1)

dados = dados.drop('Unnamed: 14', 1)

dados = dados.drop('Unnamed: 15', 1)

dados = dados.drop('Unnamed: 16', 1)
dados.head()
dados.info()
dados['Valor_total'] = dados['Valor_total'].astype("float")

dados['Media_Valor'] = dados['Media_Valor'].astype("float")

dados['Ano'] = dados['Ano'].astype("float")

dados['QTD'] = 1

dados.info()
dados[dados.duplicated(keep=False)]
dados = dados.rename(columns={'Pos.': 'Posicao'})
dados.info()
dados['GolsF/S']
divisao=dados['GolsF/S'].str.split(':')
divisao.head()
GolsFeitos = divisao.str.get(0)

GolsSofridos = divisao.str.get(1)
dados['GolsFeitos']=GolsFeitos

dados['GolsSofridos']=GolsSofridos
dados['GolsFeitos'] = dados['GolsFeitos'].astype("int")

dados['GolsSofridos'] = dados['GolsSofridos'].astype("int")

dados.head()
dados.info()
dados = dados.drop('GolsF/S', 1)
dados.head()
dados['Ano'] = dados['Ano']+1
dados
clubes_vencedores=dados[dados['Posicao']==1]

clubes_vencedores1=dados[dados['Posicao']==1]



clubes_vencedores
clubes_vencedores=clubes_vencedores.groupby('Clubes')['Posicao'].sum().reset_index()

clubes_vencedores=clubes_vencedores.sort_values(by='Posicao', ascending=False)



clubes_vencedores1=clubes_vencedores1.groupby('Clubes')['Valor_total'].sum().reset_index()

clubes_vencedores1=clubes_vencedores1.sort_values(by='Valor_total', ascending=False)

plt.figure(figsize=(15,5))

sns.barplot(x='Clubes', y='Posicao', data=clubes_vencedores)
sns.barplot(x='Clubes', y='Valor_total', data=clubes_vencedores1)
clubes_valores1=dados

clubes_valores=dados

saldo_gols=dados

gols_times=dados

vice=dados

vice=vice[vice['Posicao']==2]

participacao=dados



#clubes_valores=dados[dados['Posicao']==1]

clubes_valores1=clubes_valores1.groupby('Clubes')['Valor_total'].sum().reset_index()

clubes_valores1=clubes_valores1.sort_values(by='Valor_total', ascending=False)

clubes_valores=clubes_valores.groupby('Clubes')['Valor_total'].sum().reset_index()

clubes_valores=clubes_valores.sort_values(by='Valor_total', ascending=False)

saldo_gols=saldo_gols.groupby('Ano')['GolsFeitos'].sum().reset_index()

saldo_gols=saldo_gols.sort_values(by='GolsFeitos', ascending=False)

gols_times=gols_times.groupby('Clubes')['GolsFeitos'].sum().reset_index()

gols_times=gols_times.sort_values(by='GolsFeitos', ascending=False)

vice=vice.groupby('Clubes')['QTD'].sum().reset_index()

vice=vice.sort_values(by='QTD', ascending=False)

participacao=participacao.groupby('Clubes')['QTD'].sum().reset_index()

participacao=participacao.sort_values(by='QTD', ascending=False)
%matplotlib inline



plt.figure(figsize=(20,7))

plt.xticks(rotation=65)

sns.barplot(x='Clubes', y='Valor_total', data=clubes_valores1)
plt.figure(figsize=(10,5))

sns.pointplot(x='Ano', y='GolsFeitos', data=saldo_gols, color='blue')

plt.title('Gols por Edição')

plt.xticks(rotation=65)

plt.locator_params(axis='y', nbins=15)

plt.grid(True, color='grey')

plt.show()
%matplotlib inline



plt.figure(figsize=(20,7))

plt.xticks(rotation=65)

sns.barplot(x='Clubes', y='GolsFeitos', data=gols_times)
plt.figure(figsize=(20,7))

plt.xticks(rotation=65)

sns.barplot(x='Clubes', y='QTD', data=vice)
plt.figure(figsize=(20,7))

plt.xticks(rotation=65)

sns.barplot(x='Clubes', y='QTD', data=participacao)