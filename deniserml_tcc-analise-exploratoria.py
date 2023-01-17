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
#Importando as Bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Importando o arquivo

#Importing File

file = "../input/COTAHIST_A2009_to_A2018P.csv"

df = pd.read_csv(file)

df.head(10)
VARIAVEIS = [["TIPREG", "Quantitativa Discreta", "Tipo de Registro, sempre 1"],["DATPRE","Quantitativa Discreta", "Data de realização do pregão"],["CDOBDI", "Qualitativa Nominal", "Código de classificação do tipo do papel emitido"],

            ["CODNEG", "Qualitativa Nominal", "Código da Negociação"],["TPMERC","Qualitativa Nominal", "Código do mercado em que o papel está cadastrado"],["NOMRES","Qualitativa Nominal", "Nome resumido da empresa emissora do papel"],

            ["ESPECI", "Qualitativa Nominal", "Especificação do tipo de papel"],["PRAZOT","Quantitativa Discreta", "Prazo em dias do mercado a termo"],["MODREF","Qualitativa Nominal", "Moeda de referência"],

            ["PREABE", "Quantitativa Contínua", "Preço de abertura do papel no pregão"],["PREMAX","Quantitativa Contínua", "Preço máximo do papel no pregão"],["PREMIN","Quantitativa Contínua", "Preço mínimo do papel no pregão"],

            ["PREMED", "Quantitativa Contínua", "Preço médio do papel no pregão"],["PREULT","Quantitativa Contínua", "Preço do papel no fechamento do pregão"],["PREOFC","Quantitativa Contínua", "Preço da melhor oferta de compra pelo papel no mercado"],

            ["PREOFV", "Quantitativa Contínua", "Preço da melhor oferta de venda no mercado"],["TOTNEG","Quantitativa Discreta", "Total de negociações realizadas com o papel no pregão"],["QUATOT","Quantitativa Discreta", "Quantidade de títulos negociados nesse papel no mercado"],

            ["VOLTOT", "Quantitativa Discreta", "Volume total de titulos negociados nesse papel no mercado"],["PREEXE","Quantitativa Contínua", "Preço de exercício para o mercado de opções ou de termo secundario"],["INDOPC","Qualitativa Nominal", "Indicador de correção de preços de exercícios ou valores de contrato para os mercados de opções ou termo de secundário"],

            ["DATVEN", "Quantitativa Discreta", "Data de vencimento para os mercados de opções e de termo secundário"],["FATCOT", "Quantitativa Discreta", "Fator de cotação do papel"],["PTOEXE","Quantitativa Discreta", "Preço de exercício em pontos para opções referenciadas em dólar ou valor de contrato em pontos para termo secúndario"],

           ["CODISI", "Qualitativa Nominal", "Código do papel no sistema ISIN ou código interno do papel"],["DISMES","Quantitativa Discreta", "Número de distribuição do papel"]] 

DF_VARIAV = pd.DataFrame(VARIAVEIS, columns=["Variavel", "Classificação", "Descrição"])

DF_VARIAV
#linhas e colunas

n_lin = len(df.index)

n_col = len(df.columns)



print('Linhas',n_lin,'Colunas', n_col)
#Dados faltantes

nan_values = df.isna().sum()

nan_values[nan_values > 0]
#Quais as 5 ações mais negociadas no mercado de opções? 

stg = df[['CODNEG','TPMERC']]

mask = ((stg['TPMERC'] == 12) | (stg['TPMERC'] == 13) | (stg['TPMERC'] == 70) | (stg['TPMERC'] == 80))



by_cod = stg[mask]

by_cod = by_cod.groupby('CODNEG').size().sort_values(ascending=False)

by_cod = by_cod.head(5)



by_cod.plot(kind = 'barh', color='gray',figsize=(9,8),grid = False)



plt.title('AS 10 AÇÕES MAIS NEGOCIADAS NO MERCADO DE OPÇÕES')

plt.xlabel('TRANSAÇÕES')

plt.ylabel('AÇÕES')

plt.show()
def grafico_1():

    #Pegando a maior ação negociada no mercado de opções, qual a operação mais recorrente para esta no mercado?

    #Compra ou e venda da opção?



    #Mercados

    #012 EXERCÍCIO DE OPÇÕES DE COMPRA

    #013 EXERCÍCIO DE OPÇÕES DE VENDA

    #070 OPÇÕES DE COMPRA

    #080 OPÇÕES DE VENDA



    stg = df[['CODNEG','TPMERC']]

    mask = ((stg['TPMERC'] == 12) | (stg['TPMERC'] == 13) | (stg['TPMERC'] == 70) | (stg['TPMERC'] == 80))



    by_cod = stg[mask]

    mask1 =(stg['CODNEG'] == OPCAO_DESEJADA)

    by_market = by_cod[mask1]



    by_market = by_market.groupby(['TPMERC']).size()

    print(by_market)
def grafico_2():

    #Número de Transações por ano

    stg = df[['CODNEG','DATPRE','TPMERC']]

    stg['ANO'] = pd.to_datetime(stg.DATPRE).dt.year

    mask = ((stg['TPMERC'] == 12) | (stg['TPMERC'] == 13) | (stg['TPMERC'] == 70) | (stg['TPMERC'] == 80))



    by_year = stg[mask]

    mask1 =(stg['CODNEG'] == OPCAO_DESEJADA)

    by_year = by_year[mask1]

    by_year = by_year.groupby(['ANO']).size()



    by_year.plot(kind = 'bar', color='green',figsize=(9,10),grid = False)



    plt.xticks(rotation='90')

    plt.title('NÚMERO DE TRANSAÇÕES POR ANO')

    plt.xlabel('ANO')

    plt.ylabel('QUANTIDADE')

    print(by_year)

    plt.show()
def grafico_3():

#Preço médio da ação por ano



    stg = df[['CODNEG','TPMERC','PREEXE','DATPRE']]

    mask = ((stg['TPMERC'] == 12) | (stg['TPMERC'] == 13) | (stg['TPMERC'] == 70) | (stg['TPMERC'] == 80))

    stg['ANO'] = pd.to_datetime(stg.DATPRE).dt.year



    by_price = stg[mask]

    mask1 =(stg['CODNEG']== OPCAO_DESEJADA)

    by_price = by_price[mask1]

    by_price = by_price.groupby('ANO')['PREEXE'].mean().round(2)



    by_price.plot(color = 'purple',figsize=(12,5),grid = True)



    plt.xlabel('ANO DE REFERÊNCIA')

    plt.ylabel('PREÇO MÉDIO DE EXERICIO')

    plt.show()
def grafico_4():

    #Preço médio

    fields = ['DATPRE', 'CODNEG', 'PREABE', 'PREMIN', 'PREULT', 'PREMAX']

    df2 = pd.read_csv(

        '../input/COTAHIST_A2009_to_A2018P.csv', 

        usecols=fields

    )
def grafico_5(): 

    # Primeiros valores negociados - opções

    df_filter = df[df.CODNEG == OPCAO_DESEJADA]

    df_filter = df_filter[(df_filter['DATPRE'] >= '2009-01-01') & (df_filter['DATPRE'] <= '2009-02-01')]

    df_filter
def grafico_6():

    # Ultimos valores negociados - opções

    df_filter = df[df.CODNEG == OPCAO_DESEJADA]

    df_filter = df_filter[(df_filter['DATPRE'] >= '2017-12-01') & (df_filter['DATPRE'] <= '2018-01-01')]

    df_filter
def conteudos():

    print(OPCAO_DESEJADA)



    grafico_1()

    grafico_2()

    grafico_3()

    grafico_4()

    grafico_5()

    grafico_6()
ask = True

print("Insira o código da opção desejada ou sair para finalizar: ")

while(ask==True):

    opcao = input()

    opcao = opcao.upper()

    

    stg = df[['CODNEG','TPMERC']]

    results = stg[stg['CODNEG'] == opcao]

    

    if(opcao != 'SAIR'):

        if(len(results) > 0):

            OPCAO_DESEJADA = opcao

            ask = False 

            conteudos()

        else:

            print("Opcação não encontrada, digite novamente ou sair para finalizar.", end="")

      

    else:

        ask = False

        print("Obrigado! Até a Próxima!")

        OPCAO_DESEJADA = 0