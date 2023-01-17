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



# Any results you write to the current directory are saved as output.
# importando as bibliotecas

import pandas as pd

import seaborn as sns
# criando o dataframe

# determinando que o separador é ';'



df = pd.read_csv('/kaggle/input/Pesquisadores.csv', sep=';')
# verificar as 5 primeiras linhas para saber se a importação ocorreu corretamente

df.head()
# verificando se todas as linhas foram importadas

df.tail()
# verificando se os títulos das colunas estão corretos

df.columns
# fazer a contagem do número de registros

df.count()
# para fazer uma análise exploratória da base

df.describe()
# verificar o total de pesquisadores, se há algum país que tenha o mesmo número que outro pais

pd.value_counts(df['Total de Pesquisadores 2017'])
# localizar alguma informação específica: qual país tem a população de 8.795.070 habitantes?

df.loc[df['Populacao 2017']==8795070]
# localizar quais países tem População entre 10 e 20 milhões de habitantes

df.loc[(df['Populacao 2017']>= 10000000) & (df['Populacao 2017'] < 20000000)]
# se quiser ordenar a pesquisa por população, da maior para a menor

df.sort_values(by='Populacao 2017', ascending=False)
# se quiser ordenar a pesquisa por população, da menor para a maior

df.sort_values(by='Populacao 2017', ascending=True)
# criando uma nova coluna definindo o percentual de pesquisadores de cada população



df['Percentual de Pesquisadores'] = (df['Total de Pesquisadores 2017']/ df['Populacao 2017']*100)
# verificar se os cálculos ficou correto

df.head()
# localizar quais países tem percentual de pesquisadores maior que 0,5%

df.loc[(df['Percentual de Pesquisadores']>= 0.50)]
# ordenando os 5 países com maior percentual de pesquisadores



df.sort_values(by='Percentual de Pesquisadores', ascending=False)[0:5]
# identificando o menor índice percentual de pesquisadores



df['Percentual de Pesquisadores'].min()
# identificando o maior índice percentual de pesquisadores



df['Percentual de Pesquisadores'].max()
# criar uma categoria para saber se o percentual de pesquisadores é baixo ou alto

def categoria(s):

    if s >= 0.5:

       return 'alto'

    elif s < 0.5:

       return 'baixo'
# criar uma nova coluna dizendo se a quantidade percentual de pesquisadores em relação a população é alto ou baixo



df['cat_perc'] = df['Percentual de Pesquisadores'].apply(categoria)
# verificando os resultados

df.head()
# fazer a contagem do total de países pesquisados pela OCDE onde o percentual de pesquisadores é baixo em relação a 0,5% ou alto em relação ao mesmo valor



pd.value_counts(df['cat_perc'])
# verificando se tem algum dado faltando na base de dados

df.isnull().sum()
# fazendo os primeiros gráficos



df['Populacao 2017'].plot()
# Fazendo um gráfico do tipo Scatter, cruzando o número de pesquisadores em relação ao total de população do país, em cor vermelha

df.plot(x='Populacao 2017',y='Total de Pesquisadores 2017',kind='scatter', title='Populacao x Pesquisadores',color='r')
# Fazendo um gráfico do tipo Scatter, cruzando o número de pesquisadores em relação a categoria baixa ou alta, em cor amarela

df.plot(x='Percentual de Pesquisadores',y='Populacao 2017',kind='scatter', title='Percentual de Pesquisadores x Populacao',color='y')
# para fazer uma análise exploratória da base com as novas colunas

df.describe()
#verificando a simetria dos dados do dataset (mais próximo de zero, mais simétrico)

df.skew()
# fazendo um gráfico boxplot na coluna de percentual de pesquisadores



df.boxplot(column='Percentual de Pesquisadores')
# fazendo um gráfico boxplot na coluna de total de pesquisadores



df.boxplot(column='Total de Pesquisadores 2017')
# fazendo um boxplot cruzando as informações de populacao e total de pesquisadores



df.boxplot(column='Populacao 2017', by= 'Total de Pesquisadores 2017')
# existe correlação entre o Total de Pesquisadores e o Total de População? Se estiver próximo de zero não tem correlação



df.corr()
# plotando as correlações da análise





# plotando as correlações da análise



df[['Percentual de Pesquisadores','Populacao 2017']].corr().plot()
df[['Total de Pesquisadores 2017','Empregados da Ind. 2017']].corr().plot()