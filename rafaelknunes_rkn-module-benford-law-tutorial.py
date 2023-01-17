import rkn_module_benford_law as rkn_benford
import sys

import os

from IPython.display import HTML, display

from IPython.display import Image

import pandas as pd

import numpy as np

import random

from decimal import Decimal

import matplotlib.image as mpimg

from matplotlib import pyplot as plt

maxInt = sys.maxsize
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide1.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide2.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide3.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide4.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide5.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide6.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide7.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide8.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide9.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/Slide10.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(10,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/first_digit_exp.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(12,6), dpi=300)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/chi-sq-1.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(12,7), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/chi-sq-2.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
print(rkn_benford.readme())
# Create city names list with 8 cities.

city_list = ["city A", "city B", "city C", "city D", "city E", "city F", "city G", "city H"]
# Create the data set with 1000 lessons (rows)

data_app_1 = pd.DataFrame()

sizeOfDataFrame = 1000

for currentLine in range(sizeOfDataFrame):

    data_app_1 = data_app_1.append(pd.DataFrame({"city_expenditure":np.random.randint(999, size=1),"city_name":random.choice(city_list)},index=[0]))

data_app_1.reset_index(inplace = True)    

data_app_1.drop('index',axis=1,inplace=True)
# Data head

data_app_1.head(-5)
# Data types

data_app_1.dtypes
# Getting hints (0 for disaggregated analysis)

print(rkn_benford.hints(data_app_1, 0))
table_results = rkn_benford.benford(data_app_1, 0, 10, 100, 100, 100, 2, 2, "output_app1.xlsx", "")
# Order by city name

table_results[0].sort_values(by=['units'], inplace=True)
# Format table values

results_d1 = table_results[0].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(25,15), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D1__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D1_city B.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D1_city G.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_results[1].sort_values(by=['units'], inplace=True)
# Format table values

results_d2 = table_results[1].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D2__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D2_city G.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D2_city A.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_results[2].sort_values(by=['units'], inplace=True)
# Format table values

results_d3 = table_results[2].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D3__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D3_city E.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D3_city G.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Path to the database

# path = "../input/inputbenfordtutorial/sample_city_SPBR_2010.xlsx"

path = "../input/inputbenfordtutorial/sample_city_SPBR_2010.xlsx"

data_original = pd.read_excel(path, encoding='UTF-8', delimiter=';')

data_app_2 = data_original.copy()
# Data head

data_app_2.head(-5)
# Some descriptive analysis

data_app_2['unidade'].value_counts()
# Checking Data types (must be one text and one numeric)

data_app_2.dtypes
# Getting hints (0 for disaggregated analysis)

rkn_benford.hints(data_app_2, 0)
table_results_real = rkn_benford.benford(data_app_2, 0, 4, 150, 150, 150, 2, 2, "output_app_2.xlsx", "")
# Order by city name

table_results_real[0].sort_values(by=['chi_sq 4 rounds'], inplace=True, ascending=False)
# Format table values

results_d1_real = table_results_real[0].head(5).style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/real_D1__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D1_Torrinha.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D1_Brauna.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_results_real[1].sort_values(by=['chi_sq 4 rounds'], inplace=True, ascending=False)
# Format table values

results_d2_real = table_results_real[1].head(5).style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/real_D2__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D2_Roseira.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D2_Balsamo.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_results_real[2].sort_values(by=['chi_sq 4 rounds'], inplace=True, ascending=False)
# Format table values

results_d3_real = table_results_real[2].head(5).style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/real_D3__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D3_Oriente.png'))



a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/D3_Miracatu.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Cria o dataFrame com a coluna unidade e a coluna fibonacci.

total_interacoes = 1500

df_fibo = pd.DataFrame({"unit": "line", 'value': np.zeros(total_interacoes)})
# Para cada interação alimentar a linha do dataFrame com a primeira posição da lista fibonacci atualizada.



# Inicia a lista com os 3 primeiros números do fibonacci.

lista_fibo = [1,1,0]



for j in range(total_interacoes):

# inicio, contagem, passo iteração

    for i in range(2,3,1):

        lista_fibo[i] = lista_fibo[i-1] + lista_fibo[i-2]

        # print(f"interacao: {j} | Vetor fibonacci atual: {lista_fibo} ")

        # Alimenta a linha unidade

        df_fibo.iloc[j,0] = f"sequence {j}"

        # Alimenta a linha do dataFrame com a sequência atual de fibonacci.

        df_fibo.iloc[j,1] = Decimal(lista_fibo[0])

        # print(df.iloc[j,0])

        lista_fibo[0] = lista_fibo[1]

        lista_fibo[1] = lista_fibo[2]
# Checking Data types (must be one text and one numeric)

df_fibo.dtypes
data_app_3 = df_fibo.copy()

data_app_3
# Getting hints (1 for aggregated analysis)

rkn_benford.hints(data_app_3, 1)
table_results_fibo = rkn_benford.benford(data_app_3, 1, 10, 1000, 900, 800, 1, 1, "output_app_3.xlsx", "")
table_results_fibo[0]
table_results_fibo[1]
table_results_fibo[2]
fig = plt.figure(figsize=(16,10), dpi=300)



a = fig.add_subplot(1, 3, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/fibo_D1__Aggregated.png'))



a = fig.add_subplot(1, 3, 2)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/fibo_D2__Aggregated.png'))

# Fig 3

a = fig.add_subplot(1, 3, 3)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordtutorial/fibo_D3__Aggregated.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);