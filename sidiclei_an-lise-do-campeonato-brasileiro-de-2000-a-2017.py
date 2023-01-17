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
# Bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importando o arquivo CSV

partidas = pd.read_csv('../input/BRASILEIRAO.csv', encoding='iso-8859-1',delimiter =';')
# Biblioteca para gráficos interativos

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import cufflinks as cf

cf.go_offline()

# Para Notebooks

init_notebook_mode(connected=True)
# Analisando a quantidade de linhas e colunas

partidas.shape
# Analisando as informações como, tamanho, tipo, dados faltantes, etc

partidas.info()
# Olhando a quantidade de registros vazios na coluna Rodada

partidas['Rodada'].isna().value_counts()
# Olhando a quantidade de registros vazios na coluna Horario

partidas['Horario'].isna().value_counts()
# Analisando dados estatísticos das variáveis

partidas.describe()
# Analisando o cabeçalho - 3 primeiras linhas

partidas.head(3)
# Analisando os nomes das colunas 

partidas.columns
# CLUBE 1 = MANDANTE

partidas['Clube 1'].sort_values().unique()
# CLUBE 2 = VISITANTE

partidas['Clube 2'].sort_values().unique()
# VENCEDOR

partidas['Vencedor'].sort_values().unique()
# Verificando o Estado do Clube 1 MANDANTE para tratamento

partidas[partidas['Clube 1'] == 'Botafogo']['C1 Estado'].unique()
# Verificando o Estado do Clube 2 VISITANTE para tratamento

partidas[partidas['Clube 2'] == 'Botafogo']['C2 Estado'].unique()
# Verificando o Estado do Vencedor para tratamento

partidas[partidas['Vencedor'] == 'Botafogo']['Vencedor Estado'].unique()
# Verificando a quantidade de registros únicos para cada variável, e analisar as possíveis diferenças e inconsistências

partidas.nunique()
# Alterando o nome das colunas para facilitar a análise

partidas.columns = ['HORARIO', 'DIA_SEMANA', 'DATA', 'MANDANTE', 'VISITANTE', 'VENCEDOR', 'RODADA',

       'ARENA', 'GOLS_MANDANTE', 'GOLS_VISITANTE', 'UF_MANDANTE', 'UF_VISITANTE', 'UF_VENCEDOR']

partidas.columns
# Convertendo os valores das colunas MANDANTE, VISITANTE e VENCEDOR para maiusculos

# Assim corrigirá nomes duplicados, pois são case sensitive

partidas['MANDANTE'] = partidas['MANDANTE'].str.upper()

partidas['VISITANTE'] = partidas['VISITANTE'].str.upper()

partidas['VENCEDOR'] = partidas['VENCEDOR'].str.upper()
# Alterando o valor '-' da coluna VENCEDOR para 'EMPATE'

partidas['VENCEDOR'] =  partidas['VENCEDOR'].replace('-','EMPATE')
# Alterando o valor 'BOTAFOGO' das coluna MANDANTE, VISITANTE e VENCEDOR para 'BOTAFOGO-RJ

partidas['MANDANTE'] =  partidas['MANDANTE'].replace('BOTAFOGO','BOTAFOGO-RJ')

partidas['VISITANTE'] = partidas['VISITANTE'].replace('BOTAFOGO','BOTAFOGO-RJ')

partidas['VENCEDOR'] = partidas['VENCEDOR'].replace('BOTAFOGO','BOTAFOGO-RJ')
# É possível observar as variáveis MANDANTE, VISITANTE e VENCEDOR

# Veja que existia uma quantidade considerada de duplicidade devido ser case sensitive 

partidas.nunique()
# Convertendo a coluna DATA em DATETIME

partidas['DATA'] = pd.to_datetime(partidas['DATA'])
# Analisando a variável VENCEDOR, após tratamento.

partidas['VENCEDOR'].sort_values().unique()
# Criando a coluna ANO

partidas['ANO'] = partidas['DATA'].dt.year
# Criando a coluna MES

partidas['MES'] = partidas['DATA'].dt.month
# Criando a coluna NOME DO MES com as 3 primeiras letras

partidas['NOME_MES'] = pd.to_datetime(partidas['DATA']).dt.month_name().str.slice(stop=3)
# Criando a coluna DIA

partidas['DIA'] = partidas['DATA'].dt.day
# Conferindo as todas as colunas - variáveis

partidas.columns
# Criando uma lista com as colunas categóricas

colunas = [ 'DIA_SEMANA', 'MANDANTE', 'VISITANTE', 'VENCEDOR', 'RODADA',

            'UF_MANDANTE', 'UF_VISITANTE', 'UF_VENCEDOR','ANO', 'NOME_MES', 'DIA']
# Criando uma função que irá imprimir a quantidade de valores por coluna informada

def quantidade(dataset, colunas):

    for col in colunas:

        quant = dataset[col].value_counts()

        print(quant, '\n')
# Chamando a função e passando o DATASET e lista com as COLUNAS

quantidade(partidas, colunas)
# Analisando a quantidade de partidas, o total e média de gols marcados [GOLS_MANDANTES] e sofridos [GOLS_VISITANTES] dos times como MANDANTES

partidas.groupby(['MANDANTE'])[['GOLS_MANDANTE','GOLS_VISITANTE']].agg(['sum','count','mean'])
# Analisando a quantidade de partidas, o total e média de gols marcados [GOLS_VISITANTES] e sofridos [GOLS_MANDANTES] dos times como VISITANTES

partidas.groupby(['VISITANTE'])[['GOLS_MANDANTE','GOLS_VISITANTE']].agg(['sum','count','mean'])
# Analisando os dados estatísticos de GOLS_MANDANTES e GOLS_VISITANTES por ANO

partidas.groupby(['ANO'])[['GOLS_MANDANTE','GOLS_VISITANTE']].agg(['count','std', 'mean', 'max','median'])
# Visualização em PIZZA da quantidade total de GOLS_MANDANTE por ANO

partidas.iplot(kind='pie', labels='ANO',values='GOLS_MANDANTE')
# Visualização em PIZZA da quantidade total de GOLS_VISITANTE por ANO

partidas.iplot(kind='pie', labels='ANO',values='GOLS_VISITANTE')
# Criando uma matriz e analisando a quantidade de gols marcados por MANDANTES e VISITANTES separados por ANO X MES

matriz = partidas.pivot_table(values=['GOLS_MANDANTE','GOLS_VISITANTE'], 

                  index=['ANO'],   

                  columns=['MES'], 

                  aggfunc=np.sum, fill_value=0, margins=False)

#matriz.reset_index(inplace=True)

matriz
# Apresentando os dados em HEATMAP dos GOLS_MANDANTE por ANO x MES

plt.figure(figsize=(20,10))

sns.heatmap(matriz['GOLS_MANDANTE'],annot=True, fmt="d",linewidths=.7,

            cmap="YlGnBu", vmax=150,cbar_kws={"orientation": "vertical"})

plt.title('Total de GOLS dos Mandantes por ANO e MES')
# Apresentando os dados em HEATMAP dos GOLS_VISITANTE por ANO x MES

plt.figure(figsize=(20,10))

sns.heatmap(matriz['GOLS_VISITANTE'],annot=True, fmt="d",linewidths=.7,

            cmap="YlGnBu", vmax=150,cbar_kws={"orientation": "vertical"})

plt.title('Total de GOLS dos Visitantes por ANO e MES')
# Apresentado dados estatísticos de GOLS_MANDANTE e GOLS_VISITANTE

partidas.groupby('ANO')['GOLS_MANDANTE','GOLS_VISITANTE'].sum().iplot(kind='box')
# Comparando o total de GOLS_MANDANTE e GOLS_VISITANTE por ANO

partidas.groupby('ANO')['GOLS_MANDANTE','GOLS_VISITANTE'].sum().iplot(kind='bar')

# Analisando o número de vitórias de cada TIME por ANO

# Isto significa que quando for zero (0) o time não disputou a SERIE A

vencedorAno = pd.crosstab(partidas['VENCEDOR'],partidas['ANO'])

vencedorAno.reset_index(inplace=True)

vencedorAno
# Verificando os TIMES que NÃO disputaram a SERIE A em cada ANO

anoMinimo = partidas['ANO'].min()

anoMaximo = partidas['ANO'].max()

print('Times NÃO disputaram a SERIE A \n')

 

anoTime = []

lista = ''

for ano in range(anoMinimo,anoMaximo+1):

    print('ANO: ', ano)

    for time in (vencedorAno[vencedorAno[ano] == 0]['VENCEDOR']):

        lista = [ano, time]

        anoTime.append(lista)

        print(time, end=',')

    print('\n')
# Montado o dataframe com ANO e TIME que não estava disputaram a serie A

anoTime = pd.DataFrame(anoTime)

anoTime.columns = ['ANO','TIME']

anoTime
# Verificando o quantidade de ANOS que NÃO participaram da SERIE A

quant = pd.DataFrame(anoTime['TIME'].value_counts())

quant.columns = ['QUANT'] 

quant.index.names = ['TIME']

quant.reset_index(inplace=True)

quant
# A quantidade de ausências na SERIA A não significa REBAIXAMENTO

# Mas a quantidade MENOR que 4, pode ser considerada NÚMERO de REBAIXAMENTOS, devido aos TIMES serem considerados grandes

quant[quant['QUANT'] < 4].iplot(kind='pie', labels='TIME',values='QUANT')
# Criando um DATAFRAME com a quantidade de VITÓRIA de cada TIME

vencedor = pd.DataFrame(partidas[partidas['VENCEDOR'] !='EMPATE']['VENCEDOR'].value_counts())

vencedor.index.names = ['TIME']

vencedor.reset_index(inplace=True)

vencedor
# Mostrando top 10 dos TIMES que mais venceram

vencedor.nlargest(10,'VENCEDOR').iplot(kind='bar',x='TIME',y='VENCEDOR')
# Mostrando o TOP 5 dos TIMES que MENOS venceram em PIZZA

vencedor.nsmallest(5,'VENCEDOR').iplot(kind='pie', labels='TIME',values='VENCEDOR')
# Mostrando todos os vencedores

plt.figure(figsize=(18,10))

sns.barplot(x="VENCEDOR", y="TIME", data=vencedor,orient="h" )
# Criando dataframe com TIME e NUMERO DE VITORIAS por ANO

ANO_VENCEDOR = pd.DataFrame(partidas.groupby(['ANO'])['VENCEDOR'].value_counts())

# Alterando os nomes dos INDICES

ANO_VENCEDOR.index.names = ['ANO','TIME']

# Transformando os indices em COLUNAS

ANO_VENCEDOR.reset_index(inplace=True)

ANO_VENCEDOR
# Verificando os TIMES que tiveram o MAIOR e MENOR número de vitórias por ANO

anoMinimo = partidas['ANO'].min()

anoMaximo = partidas['ANO'].max()

print('MAIORES e MENORES vencedores por ANO \n')

 

lista = ''

for ano in range(anoMinimo,anoMaximo+1):

    maior = ANO_VENCEDOR[(ANO_VENCEDOR['TIME'] != 'EMPATE') & (ANO_VENCEDOR['ANO'] == ano)]['VENCEDOR'].max()

    menor = ANO_VENCEDOR[(ANO_VENCEDOR['TIME'] != 'EMPATE') & (ANO_VENCEDOR['ANO'] == ano)]['VENCEDOR'].min()

    lista = ANO_VENCEDOR[ ((ANO_VENCEDOR['ANO'] == ano) & ((ANO_VENCEDOR['VENCEDOR'] == maior) | (ANO_VENCEDOR['VENCEDOR'] == menor)))]

    print(lista, '\n')

     
# Apresentando a distribuição em relação ao número de vitórias dos times por ano

# É possível perceber que existem outliers, ou seja,  alguns times que se destacaram positivamente e outros negativamente

plt.figure(figsize=(18,10))

sns.boxplot('ANO','VENCEDOR',data=ANO_VENCEDOR[ANO_VENCEDOR['TIME'] != 'EMPATE'])
# Criando novo DATAFRAME

# Confronto = SÃO PAULO X FLAMENGO

SP_FL = pd.DataFrame(

    partidas[

        ((partidas['MANDANTE'] == 'SÃO PAULO') & (partidas['VISITANTE'] == 'FLAMENGO')) |

         ((partidas['MANDANTE'] == 'FLAMENGO') & (partidas['VISITANTE'] == 'SÃO PAULO'))       

    ]

)
SP_FL.head(3)
# Analisando a quantidade de gols marcados dos times como MANDANTES e VISITANTES

SP_FL.pivot_table(values=['GOLS_MANDANTE','GOLS_VISITANTE'], 

                  index=['MANDANTE','VISITANTE'],   

                  columns=['VENCEDOR'], 

                  aggfunc=np.sum, fill_value=0, margins=True)
# Quantidade de gols marcados pelo VENCEDOR

GOLS = SP_FL.groupby('VENCEDOR')['GOLS_MANDANTE','GOLS_VISITANTE'].sum()

GOLS.reset_index(inplace=True)

GOLS
# Apresentado os GOLS do VENCEDOR marcados como MANDANTE E VISITANTE

GOLS.iplot(kind='bar',x='VENCEDOR',y=['GOLS_MANDANTE','GOLS_VISITANTE'])
# Analisando a quantidade de mandos de campo por TIME

SP_FL.groupby('MANDANTE')['MANDANTE'].count()
# Gráfico para ilustrar número de vitórias

plt.figure(figsize=(16,4))

sns.countplot(x='VENCEDOR', data=SP_FL)

# Verificando o número total de vitórias para cada TIME

SP_FL['VENCEDOR'].value_counts()

# Verificando o números de vitórias de cada TIME como MANDANTES e VISITANTES

plt.figure(figsize=(18,10))

sns.catplot(x='VENCEDOR', data=SP_FL, hue='VISITANTE',col='MANDANTE', kind="count")
# Analisandos os resultados dos GOLS marcados pelos TIMES aprupados pelo VENCEDOR

SP_FL.groupby('VENCEDOR')['GOLS_MANDANTE'].value_counts()
# Apresentado o número de GOLS marcados pelos TIMES nos confrontos agrupados pelo VENCEDOR

plt.figure(figsize=(16,8))

sns.countplot(x='GOLS_MANDANTE', data=SP_FL, hue='VENCEDOR')
# Verificando o números de vitórias de cada TIME como MANDANTES e VISITANTES

plt.figure(figsize=(16,8))

sns.countplot(x='MANDANTE', data=SP_FL, hue='VENCEDOR')
# Criando um DATAFRAME

# Confronto entre os estados de  SÃO PAULO X RIO DE JANEIRO

SP_RJ = pd.DataFrame(

    partidas[

        ((partidas['UF_MANDANTE'] == 'SP') & (partidas['UF_VISITANTE'] == 'RJ')) |

         ((partidas['UF_MANDANTE'] == 'RJ') & (partidas['UF_VISITANTE'] == 'SP'))       

    ]

)
# Analisando a quantidade de gols marcados dos times como MANDANTES e VISITANTES

SP_RJ.pivot_table(values=['GOLS_MANDANTE','GOLS_VISITANTE'], 

                  index=['MANDANTE','VISITANTE','VENCEDOR'],                

                  aggfunc=np.sum, fill_value=0, margins=True)
# Analisando os gols MARCADOS dos times VENCEDORES como MANDANTE e VISITANTE

GOLS_SPRJ = SP_RJ.groupby('VENCEDOR')['GOLS_MANDANTE','GOLS_VISITANTE'].sum()

GOLS_SPRJ.reset_index(inplace=True)

GOLS_SPRJ
# Apresentado o resultados dos GOls

GOLS_SPRJ.iplot(kind='bar',x='VENCEDOR',y=['GOLS_MANDANTE','GOLS_VISITANTE'])
#  Verificando a quantidade de vitórios dos times de SP e RJ por ano

pd.crosstab(SP_RJ['VENCEDOR'],partidas['ANO'])
# Verificando o número de vitórias para cada

SP_RJ['VENCEDOR'].value_counts()
# Mostrando os maiores vencedores

SP_RJ[SP_RJ['VENCEDOR'] != 'EMPATE']['VENCEDOR'].iplot(kind='hist')
# Verificando qual ESTADO tem maior número de VITÓRIAS

vit = pd.DataFrame(

   SP_RJ['UF_VENCEDOR'].value_counts()

)

vit.index.names = ['ESTADO']

vit.reset_index(inplace=True)

vit
vit.iplot(kind='pie', labels='ESTADO',values='UF_VENCEDOR')
# Criando um dataframe com indice e ano para serem utilizados nos gráficos PLOT() que irá comparar vitorias de times

index_ano = pd.DataFrame(partidas['ANO'].unique())

index_ano.reset_index(inplace=True)

index_ano.columns = ['indice','ano']

index_ano
# Montando um gráfico para comparar os números de vitórias dos 4 grandes de SP por ano

plt.rcParams['figure.figsize'] = (18,7)



partidas[(partidas['UF_VENCEDOR'] == 'SP') & (partidas['VENCEDOR'] == 'SÃO PAULO')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='São Paulo',lw=5)

partidas[(partidas['UF_VENCEDOR'] == 'SP') & (partidas['VENCEDOR'] == 'SANTOS')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Santos',lw=3)

partidas[(partidas['UF_VENCEDOR'] == 'SP') & (partidas['VENCEDOR'] == 'PALMEIRAS')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Palmeiras',lw=2)

partidas[(partidas['UF_VENCEDOR'] == 'SP') & (partidas['VENCEDOR'] == 'CORINTHIANS')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Corinthians',lw=1)

plt.title('Comparando número de vitórias do 4 TIMES grandes de SP')



plt.xticks(index_ano['indice'],index_ano['ano'])

plt.ylabel('Número de vitórias')

plt.xlabel('Ano')

plt.legend()

plt.show()
# Montando um gráfico para comparar os números de vitórias dos 4 grandes de SP por ano

plt.rcParams['figure.figsize'] = (18,7)



partidas[(partidas['UF_VENCEDOR'] == 'RJ') & (partidas['VENCEDOR'] == 'FLUMINENSE')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Fluminense',lw=5)

partidas[(partidas['UF_VENCEDOR'] == 'RJ') & (partidas['VENCEDOR'] == 'BOTAFOGO-RJ')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Botafogo-RJ',lw=3)

partidas[(partidas['UF_VENCEDOR'] == 'RJ') & (partidas['VENCEDOR'] == 'VASCO')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Vasco',lw=2)

partidas[(partidas['UF_VENCEDOR'] == 'RJ') & (partidas['VENCEDOR'] == 'FLAMENGO')].groupby('ANO')['VENCEDOR'].value_counts().plot(label='Flamento',lw=1)

plt.title('Comparando número de vitórias do 4 TIMES grandes de SP')



plt.xticks(index_ano['indice'],index_ano['ano'])

plt.ylabel('Número de vitórias')

plt.xlabel('Ano')

plt.legend()

plt.show()