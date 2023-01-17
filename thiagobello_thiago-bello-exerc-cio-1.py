import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

anv = pd.read_csv("../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv", delimiter=',')



resposta = [["aeronave_operador_categoria", "Qualitativa Nominal"],["aeronave_tipo_veiculo", "Qualitativa Nominal"],["aeronave_pais_registro", "Qualitativa Nominal"],

            ["aeronave_ano_fabricacao", "Quantitativa Discreta"],["aeronave_tipo_operacao", "Qualitativa Nominal"],["aeronave_nivel_dano", "Qualitativa Nominal"],["total_fatalidades", "Quantitativa Discreta"]]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta

display(anv['aeronave_operador_categoria'].value_counts())
display(anv['aeronave_tipo_veiculo'].value_counts())
display(anv['aeronave_pais_registro'].value_counts())
display(anv['aeronave_tipo_operacao'].value_counts())
display(anv['aeronave_nivel_dano'].value_counts())
categoria_fatal =  anv[['aeronave_operador_categoria','total_fatalidades']]

categoria_fatal = categoria_fatal.loc[categoria_fatal['total_fatalidades'] > 0]

categoria_fatal = categoria_fatal.groupby(['aeronave_operador_categoria']).count()

categoria_fatal = categoria_fatal.sort_values(by=['total_fatalidades'], ascending=False)



plt.figure(figsize=(20,6))

plt.title('Total de Fatalidades por Categoria')

plt.xlabel('Categoria')

plt.ylabel('Fatalidades')

plt.bar(categoria_fatal.index, categoria_fatal.total_fatalidades)

plt.plot()
veiculo_fatal =  anv[['aeronave_tipo_veiculo','total_fatalidades']]

veiculo_fatal = veiculo_fatal.loc[veiculo_fatal['total_fatalidades'] > 0]

veiculo_fatal = veiculo_fatal.groupby(['aeronave_tipo_veiculo']).count()

veiculo_fatal = veiculo_fatal.sort_values(by=['total_fatalidades'], ascending=False)



plt.figure(figsize=(20,6))

plt.title('Total de Fatalidades por tipo de Veiculo')

plt.xlabel('Veiculo')

plt.ylabel('Fatalidades')

plt.bar(veiculo_fatal.index, veiculo_fatal.total_fatalidades)

plt.plot()
aeronave_nivel_dano = anv["aeronave_nivel_dano"].value_counts()

aeronave_nivel_dano.plot.pie(y="aeronave_nivel_dano", figsize=(20,10))
aeronave_ano_fabricacao = anv[anv['aeronave_ano_fabricacao'] > 0 ].aeronave_ano_fabricacao

aeronave_ano_fabricacao.hist(bins=50, figsize=(10,10))
pais_fatal =  anv[['aeronave_pais_registro','total_fatalidades']]

pais_fatal = pais_fatal.loc[pais_fatal['total_fatalidades'] > 0]

pais_fatal = pais_fatal.groupby(['aeronave_pais_registro']).count()

pais_fatal = pais_fatal.sort_values(by=['total_fatalidades'], ascending=False)



plt.figure(figsize=(20,6))

plt.title('Total de Fatalidades por Pais de Registro')

plt.xlabel('Pais de Registro')

plt.ylabel('Fatalidades')

plt.bar(pais_fatal.index, pais_fatal.total_fatalidades)

plt.plot()