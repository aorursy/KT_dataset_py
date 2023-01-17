import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(1)
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.ticker import FuncFormatter

import squarify
# Carregando CSV BR_eleitorado_2016_municipio

br_eletorado = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')

br_eletorado.head()
# classificação de colunas

colunas = [['uf'             ,'Qualitativa'],

           ['total_eleitores','Quantitativa'],

           ['f_16'           ,'Quantitativa'],

           ['f_17'           ,'Quantitativa'],

           ['f_18_20'        ,'Quantitativa'],

           ['f_21_24'        ,'Quantitativa'],

           ['f_25_34'        ,'Quantitativa'],

           ['f_35_44'        ,'Quantitativa'],

           ['f_45_59'        ,'Quantitativa'],

           ['f_60_69'        ,'Quantitativa'],

           ['f_70_79'        ,'Quantitativa'],

           ['f_sup_79'       ,'Quantitativa']]

resultado = pd.DataFrame(colunas, columns=["Variavel", "Classificação"])

resultado
coluna_selecionada = br_eletorado[resultado["Variavel"]].groupby('uf').sum().reset_index()

coluna_selecionada
#Funções

def format_num(numero, pos=None):

    x = 0

    while abs(numero) >= 1000:

        x += 1

        numero /= 1000.0

    return '%.0f%s' % (numero, ['', ' Mil', ' Milhões', ' Bilhões'][x])

#Dados para gráfico de barras

dados = coluna_selecionada[['uf','total_eleitores']].sort_values(by=['total_eleitores'], ascending=False)
# Gráfico

with plt.style.context('fivethirtyeight'):

    fig, ax = plt.subplots(figsize=(15, 8))

    plt.barh(y=dados['uf'], width=dados['total_eleitores'])

    plt.title('Quantidade de eleitores por estado')

    plt.xlabel('Quantidade de eleitores')

    plt.ylabel('UF')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_num))

    for i in range(len(coluna_selecionada[['uf']])):

        valor = ' {:,.0f}'.format(dados['total_eleitores'][i]).replace(',', '.')

        plt.text(x = dados['total_eleitores'][i], 

        y = dados['uf'][i],  

        s = valor, 

        size = 9,

        horizontalalignment='left', 

        verticalalignment='center')



    plt.show()
dados = coluna_selecionada[coluna_selecionada['uf']=='SP']

dados = dados.drop(columns=['uf', 'total_eleitores'])

dados = dados.transpose().reset_index()

dados.columns = ["idade", "total"]



# Definição dos elementos do gráfico

normaliza = matplotlib.colors.Normalize(vmin=min(dados.total), vmax=max(dados.total))

cor = [matplotlib.cm.Blues(normaliza(value)) for value in dados.total]



with plt.style.context('fivethirtyeight'):

    fig, ax = plt.subplots(figsize=(12, 8))

    squarify.plot(sizes=dados['total'], label=["16", "17", "18 a 20", "21 a 24", "25 a 34", "35 a 44", "45 a 59", "60 a 69", "70 a 79", "> 79"], color=cor, alpha=0.6)

    plt.title('Quantidade de eleitores do São Paulo por idade')

    plt.axis('off')

 

    plt.show()
