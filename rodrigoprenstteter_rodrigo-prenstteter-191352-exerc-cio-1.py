import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/cenipa-ocorrncias-aeronuticas-na-aviao-civil/anv.csv', delimiter=',')

df.head(1)
# Importando bibliotecas necessárias

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.ticker import FuncFormatter

import squarify
# Carregando dados existentes no CSV BR_eleitorado_2016_municipio

df_eleitores = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')

df_eleitores.head()
# Classificando colunas selecionadas

ar_colunas = [['uf'             ,'Qualitativa Nominal'],

              ['total_eleitores','Quantitativa Discreta'],

              ['f_16'           ,'Quantitativa Discreta'],

              ['f_17'           ,'Quantitativa Discreta'],

              ['f_18_20'        ,'Quantitativa Discreta'],

              ['f_21_24'        ,'Quantitativa Discreta'],

              ['f_25_34'        ,'Quantitativa Discreta'],

              ['f_35_44'        ,'Quantitativa Discreta'],

              ['f_45_59'        ,'Quantitativa Discreta'],

              ['f_60_69'        ,'Quantitativa Discreta'],

              ['f_70_79'        ,'Quantitativa Discreta'],

              ['f_sup_79'       ,'Quantitativa Discreta']]

df_classificacao = pd.DataFrame(ar_colunas, columns=["Variavel", "Classificação"])

df_classificacao
# Verificando se existem registros contendo colunas com valores não preenchidos

df_eleitores[df_classificacao["Variavel"]].isnull().sum().sum()
# Gerando tabela de frequência conforme seleção de colunas efetuada no item A

df_col_selec = df_eleitores[df_classificacao["Variavel"]].groupby('uf').sum().reset_index()

df_col_selec
# Formatador de números

def formata_numero(number, pos=None):

    magnitude = 0

    while abs(number) >= 1000:

        magnitude += 1

        number /= 1000.0

    return '%.0f%s' % (number, ['', ' Mil', ' Milhões', ' Bilhões'][magnitude])



# Definindo dados para geração de gráfico de barras

df_total_eleitores = df_col_selec[['uf','total_eleitores']].sort_values(by=['total_eleitores'], ascending=False)



# Definição dos elementos do gráfico

with plt.style.context('fivethirtyeight'):

    fig, ax = plt.subplots(figsize=(15, 8))

    plt.barh(y=df_total_eleitores['uf'], width=df_total_eleitores['total_eleitores'])

    plt.title('Número de eleitores por estado - 2016')

    plt.xlabel('Número de eleitores')

    plt.ylabel('Estado')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(formata_numero))

    for i in range(len(df_col_selec[['uf']])):

        valor = ' {:,.0f}'.format(df_total_eleitores['total_eleitores'][i]).replace(',', '.')

        plt.text(x = df_total_eleitores['total_eleitores'][i], 

        y = df_total_eleitores['uf'][i],  

        s = valor, 

        size = 9,

        horizontalalignment='left', 

        verticalalignment='center')

    ax.annotate('Fonte: Tribunal Superior Eleitoral - TSE',

                xy=(0.2, 0), xytext=(0, 0),

                xycoords=('axes fraction', 'figure fraction'),

                textcoords='offset points',

                size=10, ha='right', va='bottom')

    plt.show()
# Definindo dados para geração de gráfico de pizza

df_total_eleitores = df_col_selec[['uf','total_eleitores']].sort_values(by=['total_eleitores'], ascending=False).reset_index(drop=True).head(3)

qtd_eleitores      = df_col_selec['total_eleitores'].sum()

qtd_eleitores_top  = df_total_eleitores['total_eleitores'].sum()

qtd_eleitores      = qtd_eleitores - qtd_eleitores_top

df_total_eleitores = df_total_eleitores.assign(explode=0.0)

df_total_eleitores.loc[0, 'explode'] = 0.1

df_total_eleitores = df_total_eleitores.append({'uf': 'Outros','total_eleitores': qtd_eleitores, 'explode': 0}, ignore_index=True)



# Definição dos elementos do gráfico

with plt.style.context('fivethirtyeight'):

    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(df_total_eleitores['total_eleitores'], explode=df_total_eleitores['explode'], labels=df_total_eleitores['uf'], autopct='%1.1f%%', shadow=True,  startangle=90)

    plt.title('Maiores estados por quantidade eleitores - 2016')

    plt.axis('equal')

    ax.annotate('Fonte: Tribunal Superior Eleitoral - TSE',

                xy=(0.3, 0), xytext=(0, 0),

                xycoords=('axes fraction', 'figure fraction'),

                textcoords='offset points',

                size=10, ha='right', va='bottom')

    plt.show()
# Definindo dados para geração de gráfico de área

df_total_eleitores = df_col_selec[df_col_selec['uf']=='SP']

df_total_eleitores = df_total_eleitores.drop(columns=['uf', 'total_eleitores'])

df_total_eleitores = df_total_eleitores.transpose().reset_index()

df_total_eleitores.columns = ["idade", "total"]



# Definição dos elementos do gráfico

norm = matplotlib.colors.Normalize(vmin=min(df_total_eleitores.total), vmax=max(df_total_eleitores.total))

cores = [matplotlib.cm.Blues(norm(value)) for value in df_total_eleitores.total]



with plt.style.context('fivethirtyeight'):

    fig, ax = plt.subplots(figsize=(12, 8))

    squarify.plot(sizes=df_total_eleitores['total'], label=["16", "17", "18 a 20", "21 a 24", "25 a 34", "35 a 44", "45 a 59", "60 a 69", "70 a 79", "> 79"], color=cores, alpha=0.6)

    plt.title('Eleitores de São Paulo separados por idade - 2016')

    plt.axis('off')

    ax.annotate('Fonte: Tribunal Superior Eleitoral - TSE',

                xy=(0.2, 0), xytext=(0, 0),

                xycoords=('axes fraction', 'figure fraction'),

                textcoords='offset points',

                size=10, ha='right', va='bottom')    

    plt.show()