import datetime

# ULTIMA EXECUCAO:

print(f'ultima execucao: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

import numpy as np

import pandas as pd

import itertools    

import IPython

from datetime import timedelta



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

import matplotlib

import bokeh 

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.layouts import row, column

from bokeh.resources import INLINE

from bokeh.io import output_notebook

from bokeh.models import Span

from bokeh.io import output_notebook

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *

plt.style.use('seaborn-darkgrid')

output_notebook(resources=INLINE)
url_confirmados = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

url_mortes = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

url_recuperados = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

df_confirmados = pd.read_csv(url_confirmados)

df_mortes = pd.read_csv(url_mortes)

df_recuperados = pd.read_csv(url_recuperados)
df_confirmados.head()
# Separing in locals dataframes



df_confirmados_brasil = df_confirmados.loc[df_confirmados['Country/Region'] == 'Brazil'].copy()

df_confirmados_italia = df_confirmados.loc[df_confirmados['Country/Region'] == 'Italy'].copy()

df_confirmados_eua = df_confirmados.loc[df_confirmados['Country/Region'] == 'US'].copy()

df_confirmados_suecia = df_confirmados.loc[df_confirmados['Country/Region'] == 'Sweden'].copy()

#-----------------------------------------------------------------------------------------------

df_mortes_brasil = df_mortes.loc[df_mortes['Country/Region'] == 'Brazil'].copy()

df_mortes_italia = df_mortes.loc[df_mortes['Country/Region'] == 'Italy'].copy()

df_mortes_eua = df_mortes.loc[df_mortes['Country/Region'] == 'US'].copy()

df_mortes_suecia = df_mortes.loc[df_mortes['Country/Region'] == 'Sweden'].copy()

#-----------------------------------------------------------------------------------------------

df_recuperados_brasil = df_recuperados.loc[df_recuperados['Country/Region'] == 'Brazil'].copy()

df_recuperados_italia = df_recuperados.loc[df_recuperados['Country/Region'] == 'Italy'].copy()

df_recuperados_eua = df_recuperados.loc[df_recuperados['Country/Region'] == 'US'].copy()

df_recuperados_suecia = df_recuperados.loc[df_recuperados['Country/Region'] == 'Sweden'].copy()
df_confirmados_brasil.head()
df_confirmados_italia.head()
df_confirmados_eua.head()
df_confirmados_suecia.head()
# Fazendo com que todos os df comecem apenas 1 dia antes do 1º casi de covid 

df_confirmados_brasil = df_confirmados_brasil.iloc[:,39:]

df_confirmados_eua = df_confirmados_eua.iloc[:,4:]

df_confirmados_italia = df_confirmados_italia.iloc[:,12:]

df_confirmados_suecia = df_confirmados_suecia.iloc[:,13:]

# -------------------------------------------------------------

df_mortes_brasil = df_mortes_brasil.iloc[:,39:]

df_mortes_eua = df_mortes_eua.iloc[:,4:]

df_mortes_italia = df_mortes_italia.iloc[:,12:]

df_mortes_suecia = df_mortes_suecia.iloc[:,13:]

# -------------------------------------------------------------

df_recuperados_brasil = df_recuperados_brasil.iloc[:,39:]

df_recuperados_eua = df_recuperados_eua.iloc[:,4:]

df_recuperados_italia = df_recuperados_italia.iloc[:,12:]

df_recuperados_suecia = df_recuperados_suecia.iloc[:,13:]
# pegando o indice das colunas:

index_brasil = list(df_confirmados_brasil.columns.values) 

index_italia = list(df_confirmados_italia.columns.values) 

index_eua = list(df_confirmados_eua.columns.values) 

index_suecia = list(df_confirmados_suecia.columns.values) 
# pegando os valores de todos df do brasil para auxiliar em construções de graficos:

casos_brasil = []

mortes_brasil = []

recuperados_brasil = []

for i in index_brasil:

    confirmados_brasil = df_confirmados_brasil[i].sum()

    casos_brasil.append(confirmados_brasil)

    

    mortos_brasil = df_mortes_brasil[i].sum()

    mortes_brasil.append(mortos_brasil)

    

    recupera_brasil = df_recuperados_brasil[i].sum()

    recuperados_brasil.append(recupera_brasil)
# realizando o mesmo processo anterior para o eua

casos_eua = []

mortes_eua = []

recuperados_eua = []



for i in index_eua:

    confirmados_eua = df_confirmados_eua[i].sum()

    casos_eua.append(confirmados_eua)

    

    mortos_eua = df_mortes_eua[i].sum()

    mortes_eua.append(mortos_eua)



    recupera_eua = df_recuperados_eua[i].sum()

    recuperados_eua.append(recupera_eua)

# realizando o mesmo processo anterior para a italia

casos_italia = []

mortes_italia = []

recuperados_italia = []



for i in index_italia:



    confirmados_italia = df_confirmados_italia[i].sum()

    casos_italia.append(confirmados_italia)

    

    mortos_italia = df_mortes_italia[i].sum()

    mortes_italia.append(mortos_italia)

    

    recupera_italia = df_recuperados_italia[i].sum()

    recuperados_italia.append(recupera_italia)

   

    

    
# realizando o mesmo processo anterior para a suecia

casos_suecia  = []

mortes_suecia  = []

recuperados_suecia  = []

for i in index_suecia:

    confirmados_suecia = df_confirmados_suecia[i].sum()

    casos_suecia.append(confirmados_suecia)

    

    mortos_suecia = df_mortes_suecia[i].sum()

    mortes_suecia.append(mortos_suecia)



    recupera_suecia = df_recuperados_suecia[i].sum()

    recuperados_suecia.append(recupera_suecia)
# transformando as datas em um numpy array de dias corridos:

dias_brasil = np.array([i for i in range(len(index_brasil))]).reshape(-1, 1)

dias_eua = np.array([i for i in range(len(index_eua))]).reshape(-1, 1)

dias_italia = np.array([i for i in range(len(index_italia))]).reshape(-1, 1)

dias_suecia = np.array([i for i in range(len(index_suecia))]).reshape(-1, 1)
futuro = 60 # essa variavel define quantos dias para frente a partir de hoje queremos prever.

# fazendo  datas futuras em numpy array de dias corridos 

futuro_brasil = np.array([i for i in range(len(index_brasil) + futuro)]).reshape(-1, 1)

futuro_eua = np.array([i for i in range(len(index_eua) + futuro)]).reshape(-1, 1)

futuro_italia = np.array([i for i in range(len(index_italia) + futuro)]).reshape(-1, 1)

futuro_suecia = np.array([i for i in range(len(index_suecia) + futuro)]).reshape(-1, 1)
# transformando datas futuras corridas em datas de formato mm/dd/yy

start_br = '2/26/20'

start_br = datetime.datetime.strptime(start_br,"%m/%d/%y")

datas_futuras_br = []

for i in range(len(futuro_brasil)):

    datas_futuras_br.append((start_br + datetime.timedelta(days=i)).strftime('%m/%d/%y'))

#---------------------------------------------------------------------------------------------------

start_us = '1/22/20'

start_us = datetime.datetime.strptime(start_us,"%m/%d/%y")

datas_futuras_us = []

for i in range(len(futuro_eua)):

    datas_futuras_us.append((start_us + datetime.timedelta(days=i)).strftime('%m/%d/%y'))

#---------------------------------------------------------------------------------------------------

start_it = '1/30/20'

start_it = datetime.datetime.strptime(start_it,"%m/%d/%y")

datas_futuras_it = []

for i in range(len(futuro_italia)):

    datas_futuras_it.append((start_it + datetime.timedelta(days=i)).strftime('%m/%d/%y'))

#---------------------------------------------------------------------------------------------------

start_su = '1/31/20'

start_su = datetime.datetime.strptime(start_su,"%m/%d/%y")

datas_futuras_su = []

for i in range(len(futuro_suecia)):

    datas_futuras_su.append((start_su + datetime.timedelta(days=i)).strftime('%m/%d/%y'))
# transpondo todos os dataframes

df_confirmados_brasil = df_confirmados_brasil.T

df_confirmados_brasil['dates'] = index_brasil



df_mortes_brasil = df_mortes_brasil.T

df_mortes_brasil['dates'] = index_brasil



df_recuperados_brasil = df_recuperados_brasil.T

df_recuperados_brasil['dates'] = index_brasil



new_index_br = [x for x in range(len(index_brasil))]

#----------------------------------------------------

df_confirmados_eua = df_confirmados_eua.T

df_confirmados_eua['dates'] = index_eua



df_mortes_eua = df_mortes_eua.T

df_mortes_eua['dates'] = index_eua



df_recuperados_eua = df_recuperados_eua.T

df_recuperados_eua['dates'] = index_eua



new_index_eua = [x for x in range(len(index_eua))]

#----------------------------------------------------

df_confirmados_italia = df_confirmados_italia.T

df_confirmados_italia['dates'] = index_italia



df_mortes_italia = df_mortes_italia.T

df_mortes_italia['dates'] = index_italia



df_recuperados_italia = df_recuperados_italia.T

df_recuperados_italia['dates'] = index_italia



new_index_it = [x for x in range(len(index_italia))]

#----------------------------------------------------

df_confirmados_suecia = df_confirmados_suecia.T

df_confirmados_suecia['dates'] = index_suecia



df_mortes_suecia = df_mortes_suecia.T

df_mortes_suecia['dates'] = index_suecia



df_recuperados_suecia = df_recuperados_suecia.T

df_recuperados_suecia['dates'] = index_suecia

new_index_su = [x for x in range(len(index_suecia))]

df_confirmados_brasil.head()

# como vimos os index estão errados, iremos arrumar-los agora.

df_confirmados_brasil['index'] = new_index_br

df_confirmados_brasil = df_confirmados_brasil.set_index('index')

df_confirmados_brasil.rename(columns={28:'cases',

                   'dates':'dates'}, 

                 inplace=True)

df_confirmados_brasil.index.name = None

df_confirmados_brasil['dates'] = dias_brasil



df_mortes_brasil['index'] = new_index_br

df_mortes_brasil = df_mortes_brasil.set_index('index')

df_mortes_brasil.rename(columns={28:'deaths',

                   'dates':'dates'}, 

                 inplace=True)

df_mortes_brasil.index.name = None

df_mortes_brasil['dates'] = dias_brasil



df_recuperados_brasil['index'] = new_index_br

df_recuperados_brasil = df_recuperados_brasil.set_index('index')

df_recuperados_brasil.rename(columns={29:'recovery',

                   'dates':'dates'}, 

                 inplace=True)

df_recuperados_brasil.index.name = None

df_recuperados_brasil['dates'] = dias_brasil



#------------------------------------------------

df_confirmados_eua['index'] = new_index_eua

df_confirmados_eua = df_confirmados_eua.set_index('index')

df_confirmados_eua.rename(columns={225:'cases',

                   'dates':'dates'}, 

                 inplace=True)

df_confirmados_eua.index.name = None

df_confirmados_eua['dates'] = dias_eua



df_mortes_eua['index'] = new_index_eua

df_mortes_eua = df_mortes_eua.set_index('index')

df_mortes_eua.rename(columns={225:'deaths',

                   'dates':'dates'}, 

                 inplace=True)

df_mortes_eua.index.name = None

df_mortes_eua['dates'] = dias_eua



df_recuperados_eua['index'] = new_index_eua

df_recuperados_eua = df_recuperados_eua.set_index('index')

df_recuperados_eua.rename(columns={225:'recovery',

                   'dates':'dates'}, 

                 inplace=True)

df_recuperados_eua.index.name = None

df_recuperados_eua['dates'] = dias_eua

#------------------------------------------------

df_confirmados_italia['index'] = new_index_it

df_confirmados_italia = df_confirmados_italia.set_index('index')

df_confirmados_italia.rename(columns={137:'cases',

                   'dates':'dates'}, 

                 inplace=True)

df_confirmados_italia.index.name = None

df_confirmados_italia['dates'] = dias_italia



df_mortes_italia['index'] = new_index_it

df_mortes_italia = df_mortes_italia.set_index('index')

df_mortes_italia.rename(columns={137:'deaths',

                   'dates':'dates'}, 

                 inplace=True)

df_mortes_italia.index.name = None

df_mortes_italia['dates'] = dias_italia



df_recuperados_italia['index'] = new_index_it

df_recuperados_italia = df_recuperados_italia.set_index('index')

df_recuperados_italia.rename(columns={131:'recovery',

                   'dates':'dates'}, 

                 inplace=True)

df_recuperados_italia.index.name = None

df_recuperados_italia['dates'] = dias_italia

#------------------------------------------------

df_confirmados_suecia['index'] = new_index_su

df_confirmados_suecia = df_confirmados_suecia.set_index('index')

df_confirmados_suecia.rename(columns={205:'cases',

                   'dates':'dates'}, 

                 inplace=True)

df_confirmados_suecia.index.name = None

df_confirmados_suecia['dates'] = dias_suecia



df_mortes_suecia['index'] = new_index_su

df_mortes_suecia = df_mortes_suecia.set_index('index')

df_mortes_suecia.rename(columns={205:'deaths',

                   'dates':'dates'}, 

                 inplace=True)

df_mortes_suecia.index.name = None

df_mortes_suecia['dates'] = dias_suecia



df_recuperados_suecia['index'] = new_index_su

df_recuperados_suecia = df_recuperados_suecia.set_index('index')

df_recuperados_suecia.rename(columns={203:'recovery',

                   'dates':'dates'}, 

                 inplace=True)

df_recuperados_suecia.index.name = None

df_recuperados_suecia['dates'] = dias_suecia
df_confirmados_brasil.head()
df_mortes_eua.head()
df_recuperados_suecia.head()
df_mortes_italia.head()
hoje = datetime.datetime.now() - timedelta(days=1)

hoje = hoje.strftime("%d/%m/%y")

pop_eua = 331002651

pop_br = 212559417

pop_it = 60461826

pop_sw = 10099265


fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [confirmados_eua,confirmados_italia,confirmados_brasil,confirmados_suecia]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Numero de casos', size = 20)

plt.title('NUMERO DE CASOS DE COVID-19 NO DIA ' + hoje, size=20)

plt.show()
print('comparação casos nos 4 paises')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2200988" data-url="https://flo.uri.sh/visualisation/2200988/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [(confirmados_eua/pop_eua)*1000000, (confirmados_italia/pop_it) *1000000, 

           (confirmados_brasil/pop_br)*1000000, (confirmados_suecia/pop_sw)*1000000]

numeros = [round(num, 0) for num in numeros]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Numero de casos', size = 20)

plt.title('NUMERO DE CASOS POR MILHÃO DE HABITANTE NO DIA ' + hoje, size=20)

plt.show()
plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, color='blue')

plt.plot(dias_eua, casos_eua, color='red')

plt.plot(dias_italia, casos_italia, color='green')

plt.plot(dias_suecia, casos_suecia, color='yellow')

plt.title('COMPARACAO DE PROGRESSAO DE CASOS EM DIFERENTES PAISES', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

conf_eua1, conf_eua2 = itertools.tee(iter(list(casos_eua)))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(list(casos_italia)))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(list(casos_brasil)))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(list(casos_suecia)))

next(conf_su2)



diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]







antes_30_eua = []

for index,conteudo in enumerate(diferenca_eua):

    if conteudo >= 30:

        antes_30_eua.append(index)

eua_day = [i for i in range(len(antes_30_eua))]



antes_30_br = []

for index,conteudo in enumerate(diferenca_brasil):

    if conteudo >= 30:

        antes_30_br.append(index)

br_day = [i for i in range(len(antes_30_br))]



antes_30_su = []

for index,conteudo in enumerate(diferenca_suecia):

    if conteudo >= 30:

        antes_30_su.append(index)

su_day = [i for i in range(len(antes_30_su))]



antes_30_it = []

for index,conteudo in enumerate(diferenca_italia):

    if conteudo >= 30:

        antes_30_it.append(index)

it_day = [i for i in range(len(antes_30_it))]

p1 = figure(plot_width=800, plot_height=550, title="Tragetoria do covid-19 logaritmica",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias apos ter aumento de 30 casos diarios'

p1.yaxis.axis_label = 'Progressao casos(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 0





p1.line(br_day, casos_brasil[18:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(br_day[-1], casos_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, casos_eua[41:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(eua_day[-1], casos_eua[-1], fill_color="white", size=5)



p1.line(su_day, casos_suecia[35:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(su_day[-1], casos_suecia[-1], fill_color="white", size=5)





p1.line(it_day, casos_italia[23:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], casos_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o', color='blue')

plt.plot(dias_suecia, casos_suecia, marker='o', color='yellow')





plt.title('COMPARACAO DE PROGRESSÃO BRASIL VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Brasil', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o', color='blue')

plt.plot(dias_italia, casos_italia, marker='o', color='green')





plt.title('COMPARACAO DE PROGRESSÃO BRASIL VS ITALIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Brasil', 'Italia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o',  color='blue')

plt.plot(dias_eua, casos_eua, marker='o',  color='red')



plt.title('COMPARACAO DE PROGRESSÃO BRASIL VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Brasil', 'EUA'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, casos_italia, marker='o', color='green')

plt.plot(dias_suecia, casos_suecia, marker='o',  color='yellow')





plt.title('COMPARACAO DE PROGRESSÃO ITALIA VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_eua, casos_eua, marker='o', color='red')

plt.plot(dias_suecia, casos_suecia, marker='o',  color='yellow')





plt.title('COMPARACAO DE PROGRESSÃO  EUA VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['EUA', 'Suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, casos_italia, marker='o',  color='green')

plt.plot(dias_eua, casos_eua, marker='o', color='red')





plt.title('COMPARACAO DE PROGRESSÃO ITALIA VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['Italia', 'Eua'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in casos_eua:

  porc = (i / pop_eua)*1000000

  eua_porc.append(porc)

for i in casos_brasil:

  porc = (i / pop_br)*1000000

  br_porc.append(porc)

for i in casos_italia:

  porc = (i / pop_it)*1000000

  it_porc.append(porc)

for i in casos_suecia:

  porc = (i / pop_sw)*1000000

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, br_porc, color='blue')

plt.plot(dias_eua, eua_porc, color='red')

plt.plot(dias_italia, it_porc, color='green')

plt.plot(dias_suecia, su_porc, color='yellow')



plt.title('PROGRESSÃO DE CASOS POR MILHÃO', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Casos por milhão', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

conf_eua1, conf_eua2 = itertools.tee(iter(list(casos_eua)))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(list(casos_italia)))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(list(casos_brasil)))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(list(casos_suecia)))

next(conf_su2)

diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]

diferenca_eua_media = np.array(diferenca_eua).mean()

diferenca_italia_media = np.array(diferenca_italia).mean()

diferenca_brasil_media = np.array(diferenca_brasil).mean()

diferenca_suecia_media = np.array(diferenca_suecia).mean()



paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [diferenca_eua_media, diferenca_italia_media, diferenca_brasil_media, diferenca_suecia_media]

numeros = [round(m, 2) for m in numeros]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(paises, numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Media de Aumento diario', size = 20)

plt.title('Media Aumento diario de confirmacoes do covid-19', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(dias_brasil[1:], diferenca_brasil, color='blue')

plt.plot(dias_eua[1:], diferenca_eua, color='red')

plt.plot(dias_italia[1:], diferenca_italia, color='green')

plt.plot(dias_suecia[1:], diferenca_suecia, color='yellow')



plt.title('AUMENTO DIARIO', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Aumento dos casos', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'Suecia'],  prop={'size': 20}, loc="upper left")

plt.xticks(size=15)



plt.yticks(size=15)

plt.show()
diferenca_eua =[int(r) - int(p) for p,r in zip(casos_eua, mortes_eua)]

diferenca_italia =[int(r) - int(p) for p,r in zip(casos_italia, mortes_italia)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(casos_brasil, mortes_brasil)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(casos_suecia, mortes_suecia)]



diferenca_eua =[int(r) - int(p) for p,r in zip(diferenca_eua, recuperados_eua)]

diferenca_italia =[int(r) - int(p) for p,r in zip(diferenca_italia, recuperados_italia)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(diferenca_brasil, recuperados_brasil)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(diferenca_suecia, recuperados_suecia)]



antes_30_eua = []

for index,conteudo in enumerate(diferenca_eua):

    if conteudo >= 30:

        antes_30_eua.append(index)

eua_day = [i for i in range(len(antes_30_eua))]



antes_30_br = []

for index,conteudo in enumerate(diferenca_brasil):

    if conteudo >= 30:

        antes_30_br.append(index)

br_day = [i for i in range(len(antes_30_br))]



antes_30_su = []

for index,conteudo in enumerate(diferenca_suecia):

    if conteudo >= 30:

        antes_30_su.append(index)

su_day = [i for i in range(len(antes_30_su))]



antes_30_it = []

for index,conteudo in enumerate(diferenca_italia):

    if conteudo >= 30:

        antes_30_it.append(index)

it_day = [i for i in range(len(antes_30_it))]
p1 = figure(plot_width=800, plot_height=550, title="Casos ativos (Totais - (recuperados+ mortos))",

             x_range=(0, 100))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias desde inicio covid'

p1.yaxis.axis_label = 'Casos ativos'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day, diferenca_brasil[13:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(br_day[-1], diferenca_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, diferenca_eua[33:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(eua_day[-1], diferenca_eua[-1], fill_color="white", size=5)



p1.line(su_day, diferenca_suecia[33:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(su_day[-1], diferenca_suecia[-1], fill_color="white", size=5)





p1.line(it_day, diferenca_italia[23:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], diferenca_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



p1 = figure(plot_width=800, plot_height=550, title="Casos ativos (Totais - (recuperados+ mortos))",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias desde inicio covid'

p1.yaxis.axis_label = 'Casos Ativos(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day, diferenca_brasil[13:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(br_day[-1], diferenca_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, diferenca_eua[33:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(eua_day[-1], diferenca_eua[-1], fill_color="white", size=5)



p1.line(su_day, diferenca_suecia[33:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(su_day[-1], diferenca_suecia[-1], fill_color="white", size=5)





p1.line(it_day, diferenca_italia[23:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], diferenca_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [mortos_eua, mortos_italia, mortos_brasil, mortos_suecia]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'])



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height , label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('nº de mortes', size = 20)

plt.title('Nº Mortes por COVID-19 até o dia: ' + hoje, size=20)

plt.show()
fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [(mortos_eua/confirmados_eua)*100, (mortos_italia/confirmados_italia) *100, 

           (mortos_brasil/confirmados_brasil)*100, (mortos_suecia/confirmados_suecia)*100]

numeros = [round(num, 2) for num in numeros]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'])



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height , label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('% LETALIDADE', size = 20)

plt.title('LETALIDADE EM % DO COVID NO DIA : ' + hoje, size=20)

plt.show()
plt.figure(figsize=(20, 9))

mort_br = [round(abs((float(p) / float(r))), 2)*100 if p != 0 and r != 0 else 0 for p,r in zip(mortes_brasil, casos_brasil)]

mort_eua = [round(abs((float(p) / float(r))), 2)*100 if p != 0 and r != 0 else 0 for p,r in zip(mortes_eua, casos_eua)]

mort_it = [round(abs((float(p) / float(r))), 2)*100 if p != 0 and r != 0 else 0 for p,r in zip(mortes_italia, casos_italia)]

mort_su = [round(abs((float(p) / float(r))), 2)*100 if p != 0 and r != 0 else 0 for p,r in zip(mortes_suecia, casos_suecia)]





plt.plot(dias_brasil, mort_br, color='blue')

plt.plot(dias_eua, mort_eua, color='red')

plt.plot(dias_italia, mort_it, color='green')

plt.plot(dias_suecia, mort_su, color='yellow')





plt.title('Evolução da taxa de mortalidade', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Taxa de mortalidade (em %)', size = 30)

plt.legend(['Brasil', 'EUA','Italia','Suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

print('comparação morte 4 paises:')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2201203" data-url="https://flo.uri.sh/visualisation/2201203/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [(mortos_eua/pop_eua)*1000000, (mortos_italia/pop_it) *1000000, 

           (mortos_brasil/pop_br)*1000000, (mortos_suecia/pop_sw)*1000000]

numeros = [round(num, 0) for num in numeros]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Numero de casos', size = 20)

plt.title('NUMERO DE MORTES POR MILHÃO DE HABITANTE NO DIA ' + hoje, size=20)

plt.show()
plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, mortes_brasil, color='blue')

plt.plot(dias_eua, mortes_eua, color='red')

plt.plot(dias_italia, mortes_italia, color='green')

plt.plot(dias_suecia, mortes_suecia, color='yellow')



plt.title('COMPARACAO DE PROGRESSAO DE MORTE EM DIFERENTES PAISES', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de mortes', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

conf_eua1, conf_eua2 = itertools.tee(iter(list(mortes_eua)))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(list(mortes_eua)))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(list(mortes_brasil)))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(list(mortes_suecia)))

next(conf_su2)



diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]







antes_5_eua = []

for index,conteudo in enumerate(diferenca_eua):

    if conteudo >= 5:

        antes_5_eua.append(index)

eua_day = [i for i in range(len(antes_5_eua))]



antes_5_br = []

for index,conteudo in enumerate(diferenca_brasil):

    if conteudo >= 5:

        antes_5_br.append(index)

br_day = [i for i in range(len(antes_5_br))]



antes_5_su = []

for index,conteudo in enumerate(diferenca_suecia):

    if conteudo >= 5:

        antes_5_su.append(index)

su_day = [i for i in range(len(antes_5_su))]



antes_5_it = []

for index,conteudo in enumerate(diferenca_italia):

    if conteudo >= 5:

        antes_5_it.append(index)

it_day = [i for i in range(len(antes_5_it))]



p1 = figure(plot_width=800, plot_height=550, title="Tragetoria de mortes do  covid-19 logaritmica",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias apos ter aumento de 5 casos diarios'

p1.yaxis.axis_label = 'Progressao casos(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 0





p1.line(br_day, mortes_brasil[24:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(br_day[-1], mortes_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, mortes_eua[47:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(eua_day[-1], mortes_eua[-1], fill_color="white", size=5)



p1.line(su_day, mortes_suecia[54:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(su_day[-1], mortes_suecia[-1], fill_color="white", size=5)





p1.line(it_day, mortes_italia[39:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], mortes_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, mortes_brasil, marker='o', color='blue')

plt.plot(dias_suecia, mortes_suecia, marker='o', color='yellow')





plt.title('COMPARACAO DE MORTES BRASIL VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de mortos', size = 30)

plt.legend(['Brasil', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, mortes_italia, marker='o', color='green')

plt.plot(dias_eua, mortes_eua, marker='o', color='red')



plt.title('COMPARACAO DE MORTES ITALIA VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de mortos', size = 30)

plt.legend(['Italia', 'Eua'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in mortes_eua:

  porc = (i / pop_eua)*1000000

  eua_porc.append(porc)

for i in mortes_brasil:

  porc = (i / pop_br)*1000000

  br_porc.append(porc)

for i in mortes_italia:

  porc = (i / pop_it)*1000000

  it_porc.append(porc)

for i in mortes_suecia:

  porc = (i / pop_sw)*1000000

  su_porc.append(porc)



plt.figure(figsize=(15, 9))

plt.plot(dias_brasil, br_porc, color='blue')

plt.plot(dias_eua, eua_porc, color='red')

plt.plot(dias_italia, it_porc, color='green')

plt.plot(dias_suecia, su_porc, color='yellow')



plt.title('PROGRESSÃO DE MORTES POR MILHÃO DE HABITANTE', size=20)

plt.xlabel('Dias desde o inicio do covid', size = 20)

plt.ylabel('mortes por milhão', size = 20)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

conf_eua1, conf_eua2 = itertools.tee(iter(list(mortes_eua[38:])))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(list(mortes_italia[22:])))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(list(mortes_brasil[20:])))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(list(mortes_suecia[40:])))

next(conf_su2)

diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]

diferenca_eua_media = np.array(diferenca_eua).mean()

diferenca_italia_media = np.array(diferenca_italia).mean()

diferenca_brasil_media = np.array(diferenca_brasil).mean()

diferenca_suecia_media = np.array(diferenca_suecia).mean()



paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [diferenca_eua_media, diferenca_italia_media, diferenca_brasil_media, diferenca_suecia_media]

numeros = [round(m, 2) for m in numeros]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(paises, numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Media de Aumento diario', size = 20)

plt.title('Media Aumento diario de Mortes do covid 19 DESDE A CONFIRMAÇÃO DA 1º MORTE', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(dias_brasil[21:], diferenca_brasil, color='blue')

plt.plot(dias_eua[39:], diferenca_eua, color='red')

plt.plot(dias_italia[23:], diferenca_italia, color='green')

plt.plot(dias_suecia[41:], diferenca_suecia, color='yellow')



plt.title('AUMENTO DIARIO DE MORTES', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Aumento dos casos', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'Suecia'],  prop={'size': 20}, loc="upper left")

plt.xticks(size=15)



plt.yticks(size=15)

plt.show()
#@markdown Codigo mto grande, dois cliques para abrir-lo

#@markdown ---

conf_eua1, conf_eua2 = itertools.tee(iter(mortes_eua))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(mortes_italia))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(mortes_brasil))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(mortes_suecia))

next(conf_su2)



diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1,

                                            conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1

                                              ,conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1,

                                              conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1,

                                              conf_su2)]



depois_5_eua = []

for index,conteudo in enumerate(diferenca_eua):

    if conteudo >= 5:

        depois_5_eua.append(index)

eua_day = [i for i in range(len(depois_5_eua))]



depois_5_br = []

for index,conteudo in enumerate(diferenca_brasil):

    if conteudo >= 5:

        depois_5_br.append(index)

br_day = [i for i in range(len(depois_5_br))]



depois_5_su = []

for index,conteudo in enumerate(diferenca_suecia):

    if conteudo >= 5:

        depois_5_su.append(index)

su_day = [i for i in range(len(depois_5_su))]



depois_5_it = []

for index,conteudo in enumerate(diferenca_italia):

    if conteudo >= 5:

        depois_5_it.append(index)

it_day = [i for i in range(len(depois_5_it))]



def moving_average(a, n=7) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n

moving_eua = moving_average(diferenca_eua[47:])

moving_br = moving_average(diferenca_brasil[24:])

moving_it = moving_average(diferenca_italia[29:])

moving_su = moving_average(diferenca_suecia[52:])
p1 = figure(plot_width=800, plot_height=550, title="Estamos conseguindo achatar  a curva?",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias apos ter aumento de 5 mortes confirmadas'

p1.yaxis.axis_label = 'Progressao mortes(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day[7:], moving_br, color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(br_day[-1], moving_br[-1], fill_color="white", size=5)



p1.line(eua_day[7:], moving_eua, color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(eua_day[-1], moving_eua[-1], fill_color="white", size=5)



p1.line(su_day[5:], moving_su, color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(su_day[-1], moving_su[-1], fill_color="white", size=5)





p1.line(it_day[6:], moving_it, color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], moving_it[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



layout = Layout(

    paper_bgcolor='lightsteelblue',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Casos e mortes Brasil até o dia:" + hoje

)



fig = go.Figure(data=[

    

    go.Bar(name='Casos'

           , x=index_brasil

           , y=casos_brasil),

    

    go.Bar(name='Mortes'

           , x=index_brasil

           , y=mortes_brasil

           , text= mortes_brasil

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)

fig.show()

layout = Layout(

    paper_bgcolor='lightsteelblue',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Casos e mortes EUA até o dia:" + hoje

)



fig = go.Figure(data=[

    

    go.Bar(name='Casos'

           , x=index_eua

           , y=casos_eua),

    

    go.Bar(name='Mortes'

           , x=index_eua

           , y=mortes_eua

           , text= mortes_eua

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()

layout = Layout(

    paper_bgcolor='lightsteelblue',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Casos e mortes Italia até o dia:" + hoje

)



fig = go.Figure(data=[

    

    go.Bar(name='Casos'

           , x=index_italia

           , y=casos_italia),

    

    go.Bar(name='Mortes'

           , x=index_italia

           , y=mortes_italia

           , text= mortes_italia

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()

layout = Layout(

    paper_bgcolor='lightsteelblue',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Casos e mortes Suécia até o dia:" + hoje

)



fig = go.Figure(data=[

    

    go.Bar(name='Casos'

           , x=index_suecia

           , y=casos_suecia),

    

    go.Bar(name='Mortes'

           , x=index_suecia

           , y=mortes_suecia

           , text= mortes_suecia

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, recuperados_brasil, marker='o', color='green')

plt.plot(dias_brasil, mortes_brasil, marker='o', color='red')





plt.title('COMPARACAO DE RECUPERADOS VS MORTES BRASIL', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['recuperados', 'mortos'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, recuperados_italia, marker='o', color='green')

plt.plot(dias_italia, mortes_italia, marker='o', color='red')





plt.title('COMPARACAO DE RECUPERADOS VS MORTES ITALIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['recuperados', 'mortos'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_eua, recuperados_eua, marker='o', color='green')

plt.plot(dias_eua, mortes_eua, marker='o', color='red')





plt.title('COMPARACAO DE RECUPERADOS VS MORTES EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['recuperados', 'mortos'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_suecia, recuperados_suecia, marker='o', color='green')

plt.plot(dias_suecia, mortes_suecia, marker='o', color='red')





plt.title('COMPARACAO DE RECUPERADOS VS MORTES SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['recuperados', 'mortos'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

#separando o target(valor a ser previsto) das datas

target_confirmado_br = df_confirmados_brasil['cases'].copy()

df_confirmados_brasil.drop('cases', axis = 1, inplace= True)



target_mortos_br = df_mortes_brasil['deaths'].copy()

df_mortes_brasil.drop('deaths', axis = 1, inplace= True)



target_recuperado_br = df_recuperados_brasil['recovery']

df_recuperados_brasil.drop('recovery', axis = 1, inplace= True)

#-------------------------------------------------------------------------

target_confirmado_eua = df_confirmados_eua['cases'].copy()

df_confirmados_eua.drop('cases', axis = 1, inplace= True)



target_mortos_eua = df_mortes_eua['deaths'].copy()

df_mortes_eua.drop('deaths', axis = 1, inplace= True)



target_recuperado_eua = df_recuperados_eua['recovery']

df_recuperados_eua.drop('recovery', axis = 1, inplace= True)

#-------------------------------------------------------------------------

target_confirmado_italia = df_confirmados_italia['cases'].copy()

df_confirmados_italia.drop('cases', axis = 1, inplace= True)



target_mortos_italia = df_mortes_italia['deaths'].copy()

df_mortes_italia.drop('deaths', axis = 1, inplace= True)



target_recuperado_italia = df_recuperados_italia['recovery']

df_recuperados_italia.drop('recovery', axis = 1, inplace= True)

#-------------------------------------------------------------------------

target_confirmado_suecia = df_confirmados_suecia['cases'].copy()

df_confirmados_suecia.drop('cases', axis = 1, inplace= True)



target_mortos_suecia = df_mortes_suecia['deaths'].copy()

df_mortes_suecia.drop('deaths', axis = 1, inplace= True)



target_recuperado_suecia = df_recuperados_suecia['recovery']

df_recuperados_suecia.drop('recovery', axis = 1, inplace= True)



#COMECAREMOS PELO BRASIL:
#Casos Brasil

X_train, X_test, y_train, y_test = train_test_split(df_confirmados_brasil, target_confirmado_br, test_size=0.1, random_state=42)

poly = PolynomialFeatures(degree=4, include_bias=False)

X_poly = poly.fit_transform(X_train)

svm = LinearRegression()

print("fitting..")

svm.fit(X_poly,y_train)
pred_caso_br = svm.predict(poly.fit_transform(futuro_brasil))
plt.figure(figsize=(20, 12))

plt.plot(dias_brasil, casos_brasil, marker='x')

plt.plot(futuro_brasil[0:len(dias_brasil)],pred_caso_br[0:len(casos_brasil)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO BRASIL CASOS CONFIRMADOS VS PREVISÃO', size=30)

plt.xlabel('Dias desde 27/02/20', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['casos confirmados', 'predição por Regressão Polinomial'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

# predicao mortes brasil

X_train, X_test, y_train, y_test = train_test_split(df_mortes_brasil, target_mortos_br, test_size=0.1, random_state=42)

poly = PolynomialFeatures(degree=5, include_bias=False)

X_poly = poly.fit_transform(X_train)

svm = LinearRegression()

print('fitting...')

svm.fit(X_poly,y_train)
pred_mortes_br = svm.predict(poly.fit_transform(futuro_brasil))
plt.figure(figsize=(20, 12))

plt.plot(dias_brasil, mortes_brasil, marker='x')

plt.plot(futuro_brasil[0:len(dias_brasil)], pred_mortes_br[0:len(mortes_brasil)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO BRASIL  MORTES CONFIRMADAS VS PREVISÃO', size=30)

plt.xlabel('Dias desde 27/02/20', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['mortes confirmadas', 'predição por regressão polinomial'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

# predicao casos EUA

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_eua, target_confirmado_eua)
pred_caso_eua = mlp.predict(futuro_eua)
plt.figure(figsize=(20, 12))

plt.plot(dias_eua, casos_eua, marker='x')

plt.plot(futuro_eua[0:len(dias_eua)],pred_caso_eua[0:len(casos_eua)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO EUA CASOS CONFIRMADOS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['casos confirmados', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_mortes_eua, target_mortos_eua)
pred_mortes_eua = mlp.predict(futuro_eua)
plt.figure(figsize=(20, 12))

plt.plot(dias_eua, mortes_eua, marker='x')

plt.plot(futuro_eua[0:len(dias_eua)],pred_mortes_eua[0:len(mortes_eua)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO EUA MORTES CONFIRMADAS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['mortes confirmados', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# predicao casos ITALIA

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=30000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_italia, target_confirmado_italia)
pred_caso_italia = mlp.predict(futuro_italia)
plt.figure(figsize=(20, 12))

plt.plot(dias_italia, casos_italia, marker='x')

plt.plot(futuro_italia[0:len(casos_italia)],pred_caso_italia[0:len(casos_italia)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO ITALIA CASOS CONFIRMADOS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['casos confirmados', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

#previsao morte italia

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=30000, shuffle=True)

print('fitting...')

mlp.fit(df_mortes_italia, target_mortos_italia)
pred_mortes_italia = mlp.predict(futuro_italia)
plt.figure(figsize=(20, 12))

plt.plot(dias_italia, mortes_italia, marker='x')

plt.plot(futuro_italia[0:len(mortes_italia)],pred_mortes_italia[0:len(mortes_italia)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO ITALIA: MORTOS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de mortos', size = 30)

plt.legend(['mortes confirmadas', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_suecia, target_confirmado_suecia)
pred_caso_suecia = mlp.predict(futuro_suecia)
plt.figure(figsize=(20, 12))

plt.plot(dias_suecia, casos_suecia, marker='x')

plt.plot(futuro_suecia[0:len(dias_suecia)], pred_caso_suecia[0:len(casos_suecia)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO SUECIA CASOS CONFIRMADOS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de casos', size = 30)

plt.legend(['casos confirmados', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_mortes_suecia, target_mortos_suecia)
pred_mortes_suecia = mlp.predict(futuro_suecia)
plt.figure(figsize=(20, 12))

plt.plot(dias_suecia, mortes_suecia,marker='x')

plt.plot(futuro_suecia[0:len(dias_suecia)],pred_mortes_suecia[0:len(mortes_suecia)], linestyle = 'dashed', color='purple')

plt.title('COMPARAÇÃO SUECIA MORTOS: CONFIRMADOS VS PREVISÃO', size=30)

plt.xlabel('Dias corridos', size = 30)

plt.ylabel('Numero de mortes', size = 30)

plt.legend(['mortes confirmadas', 'predição por MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

index = ['previsoes', 'real', 'diferenca percentual', 'diferenca bruta']

pd.set_option('display.max_columns', 250)
colunas =[i for i in datas_futuras_br]

previsto = [round(i, 0) for i in pred_caso_br]

diferenca =[int(p) - int(r) for p,r in zip(pred_caso_br, casos_brasil)]

diferenca_percentual = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(diferenca, casos_brasil) ]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_caso_br = pd.DataFrame([previsto[40:], casos_brasil[40:], diferenca_percentual_formatada[40:], diferenca[40:]],

                          columns=colunas[40:], index = index)

diferenca = np.array(diferenca[40:])

diferenca_percentual = np.array(diferenca_percentual[40:])

print(f"A media de erro bruto para casos confirmados no Brasil eh: {round(np.absolute(diferenca).mean(), 0 )}")

print(f'A media de erro percentual para casos confirmados no Brasil eh: {round(np.absolute(diferenca_percentual).mean(), 2)}%')
print('CASOS CONFIRMADOS PARA O BRASIL: ')

pd.set_option('display.max_columns', 250)

df_caso_br
colunas =[i for i in datas_futuras_br]

previsto = [round(i, 0)  for i in pred_mortes_br]

diferenca =[int(p) - int(r) for p,r in zip(pred_mortes_br, mortes_brasil)]

diferenca_percentual = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(diferenca, mortes_brasil)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_morto_br = pd.DataFrame([previsto[40:], mortes_brasil[40:], diferenca_percentual_formatada[40:], diferenca[40:]], columns=colunas[40:], index = index)

diferenca = np.array(diferenca[40:])

diferenca_percentual = np.array(diferenca_percentual[40:])

print(f'A media de erro bruto para mortes no Brasil eh: {round(np.absolute(diferenca).mean(), 0)}')

print(f'A media de erro percentual mortes no Brasil eh: {round (np.absolute(diferenca_percentual).mean() , 2)}%')
print('MORTES PARA O BRASIL: ')

df_morto_br
colunas =[i for i in datas_futuras_us]

previsto = [round(i, 0) for i in pred_caso_eua]

diferenca =[int(p) - int(r) for p,r in zip(pred_caso_eua, casos_eua)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, casos_eua)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_caso_eua = pd.DataFrame([previsto[75:], casos_eua[75:], diferenca_percentual_formatada[75:], diferenca[75:]],

                           columns=colunas[75:], index = index)

diferenca = np.array(diferenca[75:])

diferenca_percentual = np.array(diferenca_percentual[75:])

print(f'A media de erro bruto para casos confirmados nos EUA eh: {round(np.absolute(diferenca).mean(), 0 )}')

print(f'A media de erro percentual para casos confirmados nos EUA eh: {round(np.absolute(diferenca_percentual).mean(), 2)}%')
print('CASOS CONFIRMADOS PARA OS ESTADOS UNIDOS: ')

df_caso_eua
colunas =[i for i in datas_futuras_us]

previsto = [round(i, 0) for i in pred_mortes_eua]

diferenca =[int(p) - int(r) for p,r in zip(pred_mortes_eua, mortes_eua)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, mortes_eua)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_morto_eua = pd.DataFrame([previsto[75:], mortes_eua[75:], diferenca_percentual_formatada[75:], diferenca[75:]], 

                            columns=colunas[75:], index = index)

diferenca = np.array(diferenca[75:])

diferenca_percentual = np.array(diferenca_percentual[75:])

print(f'A media de erro bruto para mortes nos EUA eh: {round(np.absolute(diferenca).mean(), 0)}')

print(f'A media de erro percentual mortes nos EUA eh: {round(np.absolute(diferenca_percentual).mean(), 2)}%')
print('MORTES PARA OS ESTADOS UNIDOS: ')

df_morto_eua
colunas =[i for i in datas_futuras_it]

previsto = [round(i, 0) for i in pred_caso_italia]

diferenca =[int(p) - int(r) for p,r in zip(pred_caso_italia, casos_italia)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, casos_italia)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_caso_italia = pd.DataFrame([previsto[67:], casos_italia[67:], diferenca_percentual_formatada[67:], diferenca[67:]], 

                              columns=colunas[67:], index = index)
diferenca = np.array(diferenca[67:])

diferenca_percentual = np.array(diferenca_percentual[67:])

print(f'A media de erro bruto para casos confirmados na Italia eh: {round(np.absolute(diferenca).mean(), 0)}')

print(f'A media de erro percentual para casos confirmados na Italia eh: {round(np.absolute(diferenca_percentual).mean(), 2)}%')
print('CASOS CONFIRMADOS PARA A ITALIA: ')

df_caso_italia
colunas =[i for i in datas_futuras_it]

previsto = [round(i, 0) for i in pred_mortes_italia]

diferenca =[int(p) - int(r) for p,r in zip(pred_mortes_italia, mortes_italia)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, mortes_italia)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_morto_italia = pd.DataFrame([previsto[67:], mortes_italia[67:], diferenca_percentual_formatada[67:], diferenca[67:]], 

                               columns=colunas[67:], index = index)
diferenca = np.array(diferenca[67:])

diferenca_percentual = np.array(diferenca_percentual[67:])

print(f'A media de erro bruto para mortes na Italia eh: {round(np.absolute(diferenca).mean() ,0)}')

print(f'A media de erro percentual mortes na Italia eh: {round(np.absolute(diferenca_percentual).mean(),2)}%')
print('MORTES CONFIRMADOS DA ITALIA: ')

df_morto_italia
colunas =[i for i in datas_futuras_su]

previsto = [round(i, 0) for i in pred_caso_suecia]

diferenca =[int(p) - int(r) for p,r in zip(pred_caso_suecia, casos_suecia)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, casos_suecia)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_caso_suecia = pd.DataFrame([previsto[66:], casos_suecia[66:], diferenca_percentual_formatada[66:], diferenca[66:]],

                              columns=colunas[66:],  index = index)
diferenca = np.array(diferenca[66:])

diferenca_percentual = np.array(diferenca_percentual[66:])

print(f'A media de erro bruto para casos confirmados na Suecia eh: {round(np.absolute(diferenca).mean(), 0)}')

print(f'A media de erro percentual para casos confirmados na Suecia eh: {round(np.absolute(diferenca_percentual).mean(), 2)}%')
print('CASOS CONFIRMADOS PARA A SUECIA: ')

df_caso_suecia
colunas =[i for i in datas_futuras_su]

previsto = [round(i, 0) for i in pred_mortes_suecia]

diferenca =[int(p) - int(r) for p,r in zip(pred_mortes_suecia, mortes_suecia)]

diferenca_percentual = [(int(p) / int(r))*100 if r !=0 else 0 for p,r in zip(diferenca, mortes_suecia)]

diferenca_percentual = [round(m, 2) for m in diferenca_percentual]

diferenca_percentual_formatada = [f'{i}%' for i in diferenca_percentual]

df_morto_suecia = pd.DataFrame([previsto[66:], mortes_suecia[66:],  diferenca_percentual_formatada[66:], diferenca[66:]],

                               columns=colunas[66:], index = index)
diferenca = np.array(diferenca[66:])

diferenca_percentual = np.array(diferenca_percentual[66:])

print(f'A media de erro bruto para mortes na Suecia eh: {round(np.absolute(diferenca).mean(),0)}')

print(f'A media de erro percentual mortes na Suecia eh: {round(np.absolute(diferenca_percentual).mean(),2)}%')
print('MORTES SUECIA: ')

df_morto_suecia
plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, color='blue')

plt.plot(futuro_eua, pred_caso_eua, color='red')

plt.plot(futuro_italia, pred_caso_italia, color='green')

plt.plot(futuro_suecia, pred_caso_suecia, color='yellow')



plt.title('COMPARACAO DA PREVISAO DE CASOS EM DIFERENTES PAISES', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero previsto de  casos', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

labels = ['EUA', 'Italia', 'Brasil', 'Suecia']

atualmente = [confirmados_eua,confirmados_italia,confirmados_brasil,confirmados_suecia]

apos_40 = [pred_caso_eua[-1], pred_caso_italia[-1], pred_caso_br[-1],  pred_caso_suecia[-1]]

apos_40 = [round(m, 0) for m in apos_40]

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Atualmente')

rects2 = ax.bar(x + width/2, apos_40, width, label= datas_futuras_br[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Casos')

ax.set_title('Casos atual vs previstos')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
print('comparação previsao de casos nos 4 paises')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2208057" data-url="https://flo.uri.sh/visualisation/2208057/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
eua_porc = round(((pred_caso_eua[-1] - confirmados_eua)/ confirmados_eua)*100 , 2)

br_porc =  round(((pred_caso_br[-1] - confirmados_brasil)/ confirmados_brasil)*100, 2)

it_porc =  round(((pred_caso_italia[-1] - confirmados_italia)/ confirmados_italia)*100, 2)

su_porc =  round(((pred_caso_suecia[-1] - confirmados_suecia)/ confirmados_suecia)*100, 2)





fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Brasil', 'Italia', 'Suecia']

numeros = [eua_porc,br_porc,it_porc,su_porc]

rects = plt.bar(paises,numeros, align='center', color=['red', 'blue', 'green', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Quantos aumentou em 40 dias', size = 20)

plt.title('% aumento', size=20)

plt.show()

fig = plt.figure(figsize=(12, 7))

labels = ['EUA', 'Italia', 'Brasil', 'Suecia']



atualmente = [(confirmados_eua/pop_eua)*1000000, (confirmados_italia/pop_it) *1000000, 

           (confirmados_brasil/pop_br)*1000000, (confirmados_suecia/pop_sw)*1000000]

atualmente = [round(num, 0) for num in atualmente]

apos_40 = [(pred_caso_eua[-1]/pop_eua)*1000000, (pred_caso_italia[-1]/pop_it)*1000000,

           (pred_caso_br[-1]/ pop_br)*1000000, (pred_caso_suecia[-1]/pop_sw)*1000000]

apos_40 = [round(num, 0) for num in apos_40]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Atualmente')

rects2 = ax.bar(x + width/2, apos_40, width, label=' Previsao para o dia ' + colunas[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Casos por milhão')

ax.set_title('Casos por milhão atual vs prevista ')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
p1 = figure(plot_width=800, plot_height=550, title="Tragetoria Prevista para covid-19 logaritmica",

             y_axis_type="linear", x_range=(80,190))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias Previstos'

p1.yaxis.axis_label = 'Progressao casos(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 0





p1.line(np.squeeze(futuro_brasil)[40:], pred_caso_br[40:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(futuro_brasil[-1], pred_caso_br[-1], fill_color="white", size=5)



p1.line(np.squeeze(futuro_eua)[75:], pred_caso_eua[75:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(futuro_eua[-1], pred_caso_eua[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_suecia)[66:], pred_caso_suecia[66:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(futuro_suecia[-1], pred_caso_suecia[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_italia)[67:], pred_caso_italia[67:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(futuro_italia[-1], pred_caso_italia[-1], fill_color="white", size=5)







p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, marker='o', color='blue')

plt.plot(futuro_suecia, pred_caso_suecia, marker='o',  color='yellow')





plt.title('COMPARACAO DE PREVISAO BRASIL VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Numero  previsto de casos', size = 30)

plt.legend(['Brasil', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, marker='o',  color='blue')

plt.plot(futuro_eua, pred_caso_eua, marker='o',  color='red')



plt.title('COMPARACAO DE PREVISAO BRASIL VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao de casos', size = 30)

plt.legend(['Brasil', 'EUA'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_eua, pred_caso_eua, marker='o',  color='red')

plt.plot(futuro_italia, pred_caso_italia, marker='o',  color='green')





plt.title('COMPARACAO DE PREVISAO ITALIA VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao de casos', size = 30)

plt.legend(['Eua', 'Italia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in pred_caso_eua:

  porc = (i / pop_eua)*1000000

  eua_porc.append(porc)

for i in pred_caso_br:

  porc = (i / 217089238)*1000000

  br_porc.append(porc)

for i in pred_caso_italia:

  porc = (i / 60017348)*1000000

  it_porc.append(porc)

for i in pred_caso_suecia:

  porc = (i / 10174790)*1000000

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, br_porc, color='blue')

plt.plot(futuro_eua, eua_porc, color='red')

plt.plot(futuro_italia, it_porc, color='green')

plt.plot(futuro_suecia, su_porc, color='yellow')



plt.title('PREVISAO CASOS POR MILHÃO ', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('casos por milhão', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

conf_eua1, conf_eua2 = itertools.tee(iter(list(pred_caso_eua[75:])))

next(conf_eua2)

conf_it1, conf_it2 = itertools.tee(iter(list(pred_caso_italia[67:])))

next(conf_it2)

conf_br1, conf_br2 = itertools.tee(iter(list(pred_caso_br[40:])))

next(conf_br2)

conf_su1, conf_su2 = itertools.tee(iter(list(pred_caso_suecia[66:])))

next(conf_su2)

diferenca_eua =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

diferenca_italia =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

diferenca_brasil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

diferenca_suecia =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]

diferenca_eua_media = np.array(diferenca_eua).mean()

diferenca_italia_media = np.array(diferenca_italia).mean()

diferenca_brasil_media = np.array(diferenca_brasil).mean()

diferenca_suecia_media = np.array(diferenca_suecia).mean()



paises = ['EUA', 'Italia', 'Brasil', 'Suecia']

numeros = [diferenca_eua_media, diferenca_italia_media, diferenca_brasil_media, diferenca_suecia_media]

numeros = [round(m, 0) for m in numeros]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(paises, numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Media de Aumento diario PREVISTO', size = 20)

plt.title('Media Aumento diario de confirmacoes do covid-19', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil[76:], diferenca_brasil[35:], color='blue')

plt.plot(futuro_eua[76:], diferenca_eua, color='red')

plt.plot(futuro_italia[76:], diferenca_italia[8:], color='green')

plt.plot(futuro_suecia[76:], diferenca_suecia[9:], color='yellow')



plt.title('AUMENTO DIARIO PREVISTOS', size=30)

plt.xlabel('Dias desde o inicio da previsao', size = 30)

plt.ylabel('Aumento dos casos', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'Suecia'],  prop={'size': 20}, loc="upper left")

plt.xticks(size=15)



plt.yticks(size=15)

plt.show()
plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, color='blue')

plt.plot(futuro_eua, pred_mortes_eua, color='red')

plt.plot(futuro_italia, pred_mortes_italia, color='green')

plt.plot(futuro_suecia, pred_mortes_suecia , color='yellow')



plt.title('COMPARACAO DE PREVISAO DE MORTE EM DIFERENTES PAISES', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao de mortes', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

labels = ['EUA', 'Italia', 'Brasil', 'Suecia']

atualmente = [mortos_eua, mortos_italia, mortos_brasil, mortos_suecia]

apos_40 = [pred_mortes_eua[-1], pred_mortes_italia[-1], pred_mortes_br[-1],  pred_mortes_suecia[-1]]

apos_40 = [round(m, 0) for m in apos_40]

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Atualmente')

rects2 = ax.bar(x + width/2, apos_40, width, label= 'Previsão para :' + datas_futuras_br[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Mortes')

ax.set_title('Mortes atual vs previstos')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
print('comparação previsao de Mortes nos 4 paises')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2288388" data-url="https://flo.uri.sh/visualisation/2288388/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
eua_porc = round(((pred_mortes_eua[-1] - mortos_eua)/ mortos_eua)*100 , 2)

br_porc =  round(((pred_mortes_br[-1] - mortos_brasil)/ mortos_brasil)*100, 2)

it_porc =  round(((pred_mortes_italia[-1] - mortos_italia)/ mortos_italia)*100, 2)

su_porc =  round(((pred_mortes_suecia[-1] - mortos_suecia)/ mortos_suecia)*100, 2)





fig = plt.figure(figsize=(12, 7))

paises = ['EUA', 'Brasil', 'Italia', 'Suecia']

numeros = [eua_porc,br_porc,it_porc,su_porc]

rects = plt.bar(paises,numeros, align='center', color=['red', 'blue', 'green', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('% de aumento das mortes em 40 dias', size = 20)

plt.title('% aumento', size=20)

plt.show()
fig = plt.figure(figsize=(12, 7))

labels = ['EUA', 'Italia', 'Brasil', 'Suecia']



atualmente = [(mortos_eua/pop_eua)*1000000, (mortos_italia/pop_it) *1000000, 

           (mortos_brasil/pop_br)*1000000, (mortos_suecia/pop_sw)*1000000]

atualmente = [round(num, 0) for num in atualmente]

apos_40 = [(pred_mortes_eua[-1]/pop_eua)*1000000, (pred_mortes_italia[-1]/pop_it)*1000000,

           (pred_mortes_br[-1]/ pop_br)*1000000, (pred_mortes_suecia[-1]/pop_sw)*1000000]

apos_40 = [round(num, 0) for num in apos_40]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Atualmente')

rects2 = ax.bar(x + width/2, apos_40, width, label=' Previsao para o dia ' + colunas[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Morte por milhão')

ax.set_title('Morte por milhão atual vs prevista ')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
p1 = figure(plot_width=800, plot_height=550, title="Tragetoria Prevista para mortes de  covid-19 logaritmica",

             y_axis_type="linear", x_range=(75,190))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Dias Previstos'

p1.yaxis.axis_label = 'Progressao mortes(escala logaritmica)'

p1.xaxis.minor_tick_line_width = 0





p1.line(np.squeeze(futuro_brasil)[40:], pred_mortes_br[40:], color='#3E4CC3', 

        legend_label='Brasil', line_width=1)

p1.circle(futuro_brasil[-1], pred_mortes_br[-1], fill_color="white", size=5)



p1.line(np.squeeze(futuro_eua)[75:], pred_mortes_eua[75:], color='#F54138', 

        legend_label='Estados Unidos', line_width=1)

p1.circle(futuro_eua[-1], pred_mortes_eua[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_suecia)[66:], pred_mortes_suecia[66:], color='#DBAE23', 

        legend_label='Suecia', line_width=1)

p1.circle(futuro_suecia[-1], pred_mortes_suecia[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_italia)[67:], pred_mortes_italia[67:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(futuro_italia[-1], pred_mortes_italia[-1], fill_color="white", size=5)







p1.legend.location = "bottom_right"

output_notebook()

show(p1)



fig = plt.figure(figsize=(12, 7))

labels = ['EUA', 'Italia', 'Brasil', 'Suecia']

atualmente = [(mortos_eua/confirmados_eua)*100, (mortos_italia/confirmados_italia) *100, 

           (mortos_brasil/confirmados_brasil)*100, (mortos_suecia/confirmados_suecia)*100]

atualmente = [round(num, 2) for num in atualmente]

apos_40 = [(pred_mortes_eua[-1]/pred_caso_eua[-1])*100, (pred_mortes_italia[-1]/pred_caso_italia[-1])*100,

           (pred_mortes_br[-1]/ pred_caso_br[-1])*100, (pred_mortes_suecia[-1]/pred_caso_suecia[-1])*100]

apos_40 = [round(m, 2) for m in apos_40]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Atualmente')

rects2 = ax.bar(x + width/2, apos_40, width, label=' Previsao para o dia ' + colunas[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('% mortalidade')

ax.set_title('Mortalidade atual vs prevista em %')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
plt.figure(figsize=(20, 9))

mort_br = [round(abs((float(p) / float(r))), 2) if p != 0 and r != 0 else 0 for p,r in zip(pred_mortes_br, pred_caso_br)]

mort_eua = [round(abs((float(p) / float(r))), 2) if p != 0 and r != 0 else 0 for p,r in zip(pred_mortes_eua, pred_caso_eua)]

mort_it = [round(abs((float(p) / float(r))), 2) if p != 0 and r != 0 else 0 for p,r in zip(pred_mortes_italia, pred_caso_italia)]

mort_su = [round(abs((float(p) / float(r))), 2) if p != 0 and r != 0 else 0 for p,r in zip(pred_mortes_suecia, pred_caso_suecia)]





plt.plot(futuro_brasil, mort_br, color='blue')

#plt.plot(futuro_eua, mort_eua, color='red')

plt.plot(futuro_italia, mort_it, color='green')

plt.plot(futuro_suecia, mort_su, color='yellow')





plt.title('Evolução da taxa de mortalidade', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Taxa de mortalidade', size = 30)

plt.legend(['Brasil','Italia','suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, marker='o', color='blue')

plt.plot(futuro_suecia, pred_mortes_suecia, marker='o', color='yellow')



plt.title('COMPARACAO PREVISAO DE  MORTES BRASIL VS SUECIA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao de mortes', size = 30)

plt.legend(['Brasil', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_italia, pred_mortes_italia, marker='o',  color='green')

plt.plot(futuro_eua, pred_mortes_eua, marker='o',  color='red')



plt.title('COMPARACAO PREVISAO DE  MORTES ITALIA VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao mortes', size = 30)

plt.legend(['Italia', 'Eua'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, marker='o',  color='blue')

plt.plot(futuro_eua, pred_mortes_eua, marker='o',  color='red')



plt.title('COMPARACAO PREVISAO DE  MORTES Brasil VS EUA', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Previsao mortes', size = 30)

plt.legend(['Brasil', 'Eua'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in pred_mortes_eua:

  porc = (i / 333545145)*1000000

  eua_porc.append(porc)

for i in pred_mortes_br:

  porc = (i / 217089238)*1000000

  br_porc.append(porc)

for i in pred_mortes_italia:

  porc = (i / 60017348)*1000000

  it_porc.append(porc)

for i in pred_mortes_suecia:

  porc = (i / 10174790)*1000000

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, br_porc, color='blue')

plt.plot(futuro_eua, eua_porc, color='red')

plt.plot(futuro_italia, it_porc, color='green')

plt.plot(futuro_suecia, su_porc, color='yellow')



plt.title('PROGRESSÃO  DE MORTES PREVISTAS POR MILHÃO DE POP', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('% de mortes em relação a pop', size = 30)

plt.legend(['Brasil', 'EUA','Italia', 'suecia'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
