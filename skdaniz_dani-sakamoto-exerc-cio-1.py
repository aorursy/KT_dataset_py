import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

plt.style.use('ggplot')
anv = pd.read_csv("../input/datavizfacens20182aula1exerccio2/anv.csv")

anv.head()
#columns

list(anv.columns.values)
anv['total_fatalidades'].unique()
resposta = [["aeronave_tipo_veiculo", "Qualitativa Nominal"],

            ["aeronave_fabricante","Qualitativa Nominal"],

            ["aeronave_modelo","Qualitativa Nominal"],

            ['aeronave_pais_fabricante','Qualitativa Nominal'],

            ["aeronave_ano_fabricacao","Quantitativa discreta"],

            ["aeronave_tipo_operacao","Qualitativa Nominal"],

            ["aeronave_nivel_dano","Qualitativa Nominal"],

            ["total_fatalidades","Quantitativa discreta"],

           ] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
filtro = resposta[(resposta['Classificação'] == "Qualitativa Nominal")]

filtro



dict_tb_qualitativa = {}

for n in filtro['Variavel']:

    dict_tb_qualitativa[n] = anv[n].value_counts()

    print('\n\********************************************************')

    print(anv[n].value_counts())

    print('\n\********************************************************')
lista_tipo = []

lista_qtd = []

for i, v in dict_tb_qualitativa['aeronave_tipo_veiculo'].items():

    lista_tipo.append(i)

    lista_qtd.append(v)

    

tipo_qtd = dict_tb_qualitativa['aeronave_tipo_veiculo']

tipo = lista_tipo

qtd = lista_qtd

v = np.arange(len(tipo))



plt.figure(figsize=(15,7))

plt.barh(v, qtd, align='center', color=(0.2, 0.4, 0.6, 0.6))

plt.yticks(v, tipo, color='black')

plt.xlabel('Quantidade', color='black')

plt.ylabel('Tipo de veículo (aeronave)', color='black')

plt.title('Quantidade por Tipo de veículo', color='black')



for i, v in enumerate(qtd):

    plt.text(v+0.2, i, str(round(v, 2)), color='black', va='center')



ax = plt.gca()

ax.invert_yaxis()

ax.grid(False)

ax.set_facecolor("white")



plt.show()
n_fatalidades_tipo_aeronave = anv.groupby(['aeronave_tipo_veiculo']).agg({'total_fatalidades':'sum'}).sort_values('total_fatalidades', ascending=False).reset_index()

n_fatalidades_tipo_aeronave.rename(columns={"aeronave_tipo_veiculo": "Tipo de veiculo", "total_fatalidades": "Total de fatalidades"})
data  = n_fatalidades_tipo_aeronave

lista_tipo = data['aeronave_tipo_veiculo']

num_fatalidades = data['total_fatalidades']



tipo_qtd = lista_tipo

tipo = lista_tipo

qtd = num_fatalidades

v = np.arange(len(tipo))



plt.figure(figsize=(15,7))

plt.barh(v, qtd, align='center', color='coral')

plt.yticks(v, tipo, color='black')

plt.xlabel('Total de fatalidades', color='black')

plt.ylabel('Tipo de veículo (aeronave)', color='black')

plt.title('Total de fatalidades por Tipo de veículo', color='black')



for i, v in enumerate(qtd):

    plt.text(v+0.2, i, str(round(v, 2)), color='black', va='center')



ax = plt.gca()

ax.invert_yaxis()

ax.grid(False)

ax.set_facecolor("white")



plt.show()
n_fatalidades_fabricante = anv.groupby(['aeronave_fabricante']).agg({'total_fatalidades':'sum'}).sort_values('total_fatalidades', ascending=False).reset_index()

n_fatalidades_fabricante.rename(columns={"aeronave_fabricante": "Fabricantes", "total_fatalidades": "Total de fatalidades"})
data = n_fatalidades_fabricante.head(20)

lista_fabricante = data['aeronave_fabricante']

num_fatalidades = data['total_fatalidades']



fabricante = lista_fabricante

qtd = num_fatalidades

v = np.arange(len(fabricante))



plt.figure(figsize=(15,7))

plt.barh(v, qtd, align='center', color='red')

plt.yticks(v, fabricante, color='black')

plt.xlabel('Total de fatalidades', color='black')

plt.ylabel('Fabricantes de aeronaves', color='black')

plt.title('Total de fatalidades por Fabricante (Top 20 em fatalidades)', color='black')



for i, v in enumerate(qtd):

    plt.text(v+0.2, i, str(round(v, 2)), color='black', va='center')



ax = plt.gca()

ax.invert_yaxis()

ax.grid(False)

ax.set_facecolor("white")



plt.show()
n_fatalidades_modelo = anv.groupby(['aeronave_modelo']).agg({'total_fatalidades':'sum'}).sort_values('total_fatalidades', ascending=False).reset_index()

n_fatalidades_modelo.rename(columns={"aeronave_modelo": "Modelos", "total_fatalidades": "Total de fatalidades"})
data = n_fatalidades_modelo.head(20)

lista_modelo = data['aeronave_modelo']

num_fatalidades = data['total_fatalidades']



modelo = lista_modelo

qtd = num_fatalidades

v = np.arange(len(modelo))



plt.figure(figsize=(15,7))

plt.barh(v, qtd, align='center', color='blue')

plt.yticks(v, modelo, color='black')

plt.xlabel('Total de fatalidades', color='black')

plt.ylabel('Modelos de aeronaves', color='black')

plt.title('Total de fatalidades por Modelo (Top 20 em fatalidades)', color='black')



for i, v in enumerate(qtd):

    plt.text(v+0.2, i, str(round(v, 2)), color='black', va='center')



ax = plt.gca()

ax.invert_yaxis()

ax.grid(False)

ax.set_facecolor("white")



plt.show()
n_fatalidades_fabricante = anv.groupby(['aeronave_fabricante','aeronave_ano_fabricacao']).agg({'total_fatalidades':'sum'}).sort_values('total_fatalidades', ascending=False).reset_index()

n_fatalidades_fabricante['aeronave_ano_fabricacao'] = n_fatalidades_fabricante['aeronave_ano_fabricacao'].astype(int)

n_fatalidades_fabricante
data = n_fatalidades_fabricante.head(20)

data['fabricante_ano']=data.apply(lambda x:'%s - %s' % (x['aeronave_fabricante'],x['aeronave_ano_fabricacao']),axis=1)

data.rename(columns={"fabricante_ano": "Fabricantes_Ano","total_fatalidades": "Total de fatalidades"})
lista_fabricante_ano = data['fabricante_ano']

num_fatalidades = data['total_fatalidades']



fabricante_ano = lista_fabricante_ano

qtd = num_fatalidades

v = np.arange(len(fabricante_ano))



plt.figure(figsize=(15,7))

plt.barh(v, qtd, align='center', color='red')

plt.yticks(v, fabricante_ano, color='black')

plt.xlabel('Total de fatalidades', color='black')

plt.ylabel('Fabricantes de aeronaves e Ano', color='black')

plt.title('Total de fatalidades por Fabricante & Ano (Top 20 em fatalidades)', color='black')



for i, v in enumerate(qtd):

    plt.text(v+0.2, i, str(round(v, 2)), color='black', va='center')



ax = plt.gca()

ax.invert_yaxis()

ax.grid(False)

ax.set_facecolor("white")
lista_pais = []

lista_qtd = []

for i, v in dict_tb_qualitativa['aeronave_pais_fabricante'].items():

    lista_pais.append(i)

    lista_qtd.append(v)



qtd = lista_qtd

pais = lista_pais

y = np.arange(len(pais))

 

plt.figure(figsize=(15,7))

plt.bar(y, qtd, align='center', color=("orange"))

plt.xticks(y, pais)

plt.title('Total de aeronaves por país fabricante', color='black')

 

ax = plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=55, horizontalalignment='right')

ax.grid(False)

ax.set_facecolor("white")

plt.show()

lista_tipo_op = []

lista_qtd = []

for i, v in dict_tb_qualitativa['aeronave_tipo_operacao'].items():

    lista_tipo_op.append(i)

    lista_qtd.append(v)



qtd = lista_qtd

tipo_operacao = lista_tipo_op

y = np.arange(len(tipo_operacao))

 

plt.figure(figsize=(15,7))

plt.bar(y, qtd, align='center', color=("green"))

plt.xticks(y, tipo_operacao)

plt.title('Total por tipo de operação', color='black')

 

ax = plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=55, horizontalalignment='right')

ax.grid(False)

ax.set_facecolor("white")

plt.show()
lista_nivel_dano = []

lista_qtd = []

for i, v in dict_tb_qualitativa['aeronave_nivel_dano'].items():

    lista_nivel_dano.append(i)

    lista_qtd.append(v)



qtd = lista_qtd

nivel_dano = lista_nivel_dano

y = np.arange(len(nivel_dano))

 

plt.figure(figsize=(15,7))

plt.bar(y, qtd, align='center', color=("grey"))

plt.xticks(y, nivel_dano)

plt.title('Total por tipo de operação', color='black')

 

ax = plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=55, horizontalalignment='right')

ax.grid(False)

ax.set_facecolor("white")

plt.show()