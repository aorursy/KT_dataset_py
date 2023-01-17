import datetime

# Last execution:

print(f'Last execution: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

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
# split dataframe in locals dataframe



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
# Making all dataframes started 1 day after 1º case of covid

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
# Geting the index columns

index_brasil = list(df_confirmados_brasil.columns.values) 

index_italia = list(df_confirmados_italia.columns.values) 

index_eua = list(df_confirmados_eua.columns.values) 

index_suecia = list(df_confirmados_suecia.columns.values) 
# geting  values ​​of all brazil df to assist in the construction of graphics:

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
# making the same for us

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

# making the same for Italy

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

   

    

    
# making the same for Sweden

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
# transform dates into a numpy array of calendar days:

dias_brasil = np.array([i for i in range(len(index_brasil))]).reshape(-1, 1)

dias_eua = np.array([i for i in range(len(index_eua))]).reshape(-1, 1)

dias_italia = np.array([i for i in range(len(index_italia))]).reshape(-1, 1)

dias_suecia = np.array([i for i in range(len(index_suecia))]).reshape(-1, 1)
futuro = 40 # this variable defines how many days ahead we want to predict.



# making future dates in a calendar day numpy array

futuro_brasil = np.array([i for i in range(len(index_brasil) + futuro)]).reshape(-1, 1)

futuro_eua = np.array([i for i in range(len(index_eua) + futuro)]).reshape(-1, 1)

futuro_italia = np.array([i for i in range(len(index_italia) + futuro)]).reshape(-1, 1)

futuro_suecia = np.array([i for i in range(len(index_suecia) + futuro)]).reshape(-1, 1)
# transforming future calendar day in mm/dd/yy format

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
# transposing all dataframes

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

# as we have seen the indexes are wrong, we will fix them now.



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
today = datetime.datetime.now() - timedelta(days=1)

today = today.strftime("%d/%m/%y")

pop_eua = 331002651

pop_br = 212559417

pop_it = 60461826

pop_sw = 10099265
fig = plt.figure(figsize=(12, 7))

country = ['US', 'Italy', 'Brazil', 'Sweden']

numbers = [confirmados_eua,confirmados_italia,confirmados_brasil,confirmados_suecia]

rects = plt.bar(country,numbers, align='center', color=['red', 'green', 'blue', 'yellow'])



ax = rects.patches

for rect, label in zip(ax, numbers):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    



plt.ylabel('Number of cases per country', size = 20)

plt.title('NUMBER OF COVID-19 CASES, IN ' + today, size=20)

plt.show()
print('comparison of cases in the 4 countries')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2200988" data-url="https://flo.uri.sh/visualisation/2200988/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
fig = plt.figure(figsize=(12, 7))

paises = ['US', 'Italy', 'Brazil', 'Sweden']

numeros = [(confirmados_eua/pop_eua)*1000000, (confirmados_italia/pop_it) *1000000, 

           (confirmados_brasil/pop_br)*1000000, (confirmados_suecia/pop_sw)*1000000]

numeros = [round(num, 0) for num in numeros]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Number of cases', size = 20)

plt.title('Number of cases per million of pop in ' + today, size=20)

plt.show()
plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, color='blue')

plt.plot(dias_eua, casos_eua, color='red')

plt.plot(dias_italia, casos_italia, color='green')

plt.plot(dias_suecia, casos_suecia, color='yellow')



plt.title('COMPARISON OF CASE PROGRESSION IN DIFFERENT COUNTRIES', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





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

p1 = figure(plot_width=800, plot_height=550, title="Logarithmic covid-19 trajectory",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days after increasing 30 daily cases'

p1.yaxis.axis_label = 'Progression of cases (logarithmic scale)'

p1.xaxis.minor_tick_line_width = 0





p1.line(br_day, casos_brasil[18:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(br_day[-1], casos_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, casos_eua[41:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(eua_day[-1], casos_eua[-1], fill_color="white", size=5)



p1.line(su_day, casos_suecia[35:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(su_day[-1], casos_suecia[-1], fill_color="white", size=5)





p1.line(it_day, casos_italia[24:], color='#3EC358', 

        legend_label='Ialy', line_width=1)

p1.circle(it_day[-1], casos_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o', color='blue')

plt.plot(dias_suecia, casos_suecia, marker='o', color='yellow')





plt.title('COMPARISON OF CASE PROGRESSION BRAZIL VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Brazil', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o', color='blue')

plt.plot(dias_italia, casos_italia, marker='o', color='green')



plt.title('COMPARISON OF CASE PROGRESSION BRAZIL VS Italy', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Brazil', 'Italy'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, casos_brasil, marker='o', color='blue')

plt.plot(dias_eua, casos_eua, marker='o', color='red')



plt.title('COMPARISON OF CASE PROGRESSION BRAZIL VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Brazil', 'US'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, casos_italia, marker='o', color='green')

plt.plot(dias_suecia, casos_suecia, marker='o', color='yellow')





plt.title('COMPARISON OF CASE PROGRESSION ITALY VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_eua, casos_eua, marker='o', color='red')

plt.plot(dias_suecia, casos_suecia, marker='o', color='yellow')





plt.title('COMPARISON OF CASE PROGRESSION US VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['US', 'Sweden'],  prop={'size': 20}, loc="upper left")



plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, casos_italia, marker='o', color='green')

plt.plot(dias_eua, casos_eua, marker='o', color='red')





plt.title('COMPARISON OF CASE PROGRESSION ITALY VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Italy', 'US'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in casos_eua:

  porc = (i / 333545145)*100

  eua_porc.append(porc)

for i in casos_brasil:

  porc = (i / 217089238)*100

  br_porc.append(porc)

for i in casos_italia:

  porc = (i / 60017348)*100

  it_porc.append(porc)

for i in casos_suecia:

  porc = (i / 10174790)*100

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, br_porc, color='blue')

plt.plot(dias_eua, eua_porc, color='red')

plt.plot(dias_italia, it_porc, color='green')

plt.plot(dias_suecia, su_porc, color='yellow')



plt.title('PROGRESSION IN RELATION TO THE POPULATION', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('% of cases in relation to population', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





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

difference_us =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

difference_italy =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

difference_brazil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

difference_sweden =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]

difference_us_mean = np.array(difference_us).mean()

difference_italy_mean  = np.array(difference_italy).mean()

difference_brazil_mean = np.array(difference_brazil).mean()

difference_sweden_mean = np.array(difference_sweden).mean()





country = ['US', 'Italy', 'Brazil', 'Sweden']

numbers = [difference_us_mean, difference_italy_mean, difference_brazil_mean, difference_sweden_mean]

numbers = [round(m, 2) for m in numbers]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(country, numbers, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numbers):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Average Daily Increase', size = 20)

plt.title('Average Daily increase in covid-19 confirmation', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(dias_brasil[1:], difference_brazil, color='blue')

plt.plot(dias_eua[1:], difference_us, color='red')

plt.plot(dias_italia[1:], difference_italy, color='green')

plt.plot(dias_suecia[1:], difference_sweden, color='yellow')



plt.title('Daily Increase', size=30)

plt.xlabel('Dias desde o inicio do covid', size = 30)

plt.ylabel('Cases increase', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")

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
p1 = figure(plot_width=800, plot_height=550, title="Active cases (Total - (recovered + killed))",

             x_range=(0, 100))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days since covid start'

p1.yaxis.axis_label = 'Active cases'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day, diferenca_brasil[13:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(br_day[-1], diferenca_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, diferenca_eua[33:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(eua_day[-1], diferenca_eua[-1], fill_color="white", size=5)



p1.line(su_day, diferenca_suecia[33:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(su_day[-1], diferenca_suecia[-1], fill_color="white", size=5)





p1.line(it_day, diferenca_italia[23:], color='#3EC358', 

        legend_label='Italy', line_width=1)

p1.circle(it_day[-1], diferenca_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



p1 = figure(plot_width=800, plot_height=550, title="Active cases (Total - (recovered + killed))",

             x_range=(0, 100), y_axis_type='log')

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days since covid start'

p1.yaxis.axis_label = 'Active cases(LOG)'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day, diferenca_brasil[13:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(br_day[-1], diferenca_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, diferenca_eua[33:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(eua_day[-1], diferenca_eua[-1], fill_color="white", size=5)



p1.line(su_day, diferenca_suecia[33:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(su_day[-1], diferenca_suecia[-1], fill_color="white", size=5)





p1.line(it_day, diferenca_italia[23:], color='#3EC358', 

        legend_label='Italy', line_width=1)

p1.circle(it_day[-1], diferenca_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



fig = plt.figure(figsize=(12, 7))

country = ['US', 'Italy', 'Brazil', 'Sweden']

numbers = [mortos_eua,mortos_italia,mortos_brasil, mortos_suecia]

rects = plt.bar(country,numbers, align='center', color=['red', 'green', 'blue', 'yellow'])

ax = rects.patches

for rect, label in zip(ax, numbers):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height , label,

            ha='center', va='bottom', fontsize=15)



plt.ylabel('Numero of deads', size = 20)

plt.title('NUMBER OF COVID-19 DEATHS IN ' + today, size=20)

plt.show()
fig = plt.figure(figsize=(12, 7))

country = ['EUA', 'Italia', 'Brasil', 'Suecia']

numbers = [(mortos_eua/confirmados_eua)*100, (mortos_italia/confirmados_italia) *100, 

           (mortos_brasil/confirmados_brasil)*100, (mortos_suecia/confirmados_suecia)*100]

numbers = [round(num, 2) for num in numbers]

rects = plt.bar(country,numbers, align='center', color=['red', 'green', 'blue', 'yellow'])



ax = rects.patches

for rect, label in zip(ax, numbers):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height , label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('% Lethality', size = 20)

plt.title('LETHALY in % OF COVID in ' + today, size=20)

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





plt.title('Evolution of the mortality rate', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('mortality rate (%)', size = 30)

plt.legend(['Brazil', 'US','Italy','Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

print('death comparasion:')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2201203" data-url="https://flo.uri.sh/visualisation/2201203/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
fig = plt.figure(figsize=(12, 7))

paises = ['US', 'Italy', 'Brazil', 'Sweden']

numeros = [(mortos_eua/pop_eua)*1000000, (mortos_italia/pop_it) *1000000, 

           (mortos_brasil/pop_br)*1000000, (mortos_suecia/pop_sw)*1000000]

numeros = [round(num, 0) for num in numeros]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Number of deaths', size = 20)

plt.title('number of deaths per million in ' + today, size=20)

plt.show()
plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, mortes_brasil,color='blue')

plt.plot(dias_eua, mortes_eua, color='red')

plt.plot(dias_italia, mortes_italia, color='green')

plt.plot(dias_suecia, mortes_suecia, color='yellow')



plt.title('COMPARISON OF DEATH PROGRESSION IN DIFFERENT COUNTRIES', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['Brazil', 'US', 'Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





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



p1 = figure(plot_width=800, plot_height=550, title="Death trajectory of covid-19 logarithmic",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days after an increase of 5 daily deaths'

p1.yaxis.axis_label = 'Progression deaths (logarithmic scale)'

p1.xaxis.minor_tick_line_width = 0





p1.line(br_day, mortes_brasil[24:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(br_day[-1], mortes_brasil[-1], fill_color="white", size=5)



p1.line(eua_day, mortes_eua[47:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(eua_day[-1], mortes_eua[-1], fill_color="white", size=5)



p1.line(su_day, mortes_suecia[54:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(su_day[-1], mortes_suecia[-1], fill_color="white", size=5)





p1.line(it_day, mortes_italia[39:], color='#3EC358', 

        legend_label='Italia', line_width=1)

p1.circle(it_day[-1], mortes_italia[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, mortes_brasil, marker='o',  color='blue')

plt.plot(dias_suecia, mortes_suecia, marker='o',  color='yellow')





plt.title('COMPARISON OF DEATH PROGRESSION BRAZIL VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['Brazil', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, mortes_italia, marker='o',  color='green')

plt.plot(dias_eua, mortes_eua, marker='o',  color='red')



plt.title('COMPARISON OF DEATH PROGRESSION ITALY VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['Italy', 'US'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in mortes_eua:

  porc = (i / 333545145)*100

  eua_porc.append(porc)

for i in mortes_brasil:

  porc = (i / 217089238)*100

  br_porc.append(porc)

for i in mortes_italia:

  porc = (i / 60017348)*100

  it_porc.append(porc)

for i in mortes_suecia:

  porc = (i / 10174790)*100

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, br_porc, color='blue')

plt.plot(dias_eua, eua_porc, color='red')

plt.plot(dias_italia, it_porc, color='green')

plt.plot(dias_suecia, su_porc, color='yellow')



plt.title('PROGRESSION OF DEATHS IN RELATION TO POPULATION', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('% of deaths in relation to the population', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





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

difference_us =[int(r) - int(p) for p,r in zip(conf_eua1, conf_eua2)]

difference_italy =[int(r) - int(p) for p,r in zip(conf_it1, conf_it2)]

difference_brazil =[int(r) - int(p) for p,r in zip(conf_br1, conf_br2)]

difference_sweden =[int(r) - int(p) for p,r in zip(conf_su1, conf_su2)]

difference_us_mean = np.array(difference_us).mean()

difference_italy_mean = np.array(difference_italy).mean()

difference_brazil_mean = np.array(difference_brazil).mean()

difference_sweden_mean = np.array(difference_sweden).mean()



country = ['US', 'Italy', 'Brazil', 'Sweden']

numbers = [difference_us_mean, difference_italy_mean, difference_brazil_mean, difference_sweden_mean]

numbers = [round(m, 2) for m in numbers]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(country, numbers, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numbers):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('Average Deaths Daily Increase', size = 20)

plt.title('Average Daily increase of Deaths', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(dias_brasil[21:], difference_brazil, color='blue')

plt.plot(dias_eua[39:], difference_us, color='red')

plt.plot(dias_italia[23:], difference_italy, color='green')

plt.plot(dias_suecia[41:], difference_sweden, color='yellow')



plt.title('Dayly deaths increase', size=30)

plt.xlabel('days since covid started', size = 30)

plt.ylabel('Deaths increase', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")

plt.xticks(size=15)



plt.yticks(size=15)

plt.show()
#@markdown Very large code, two clicks to open it

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
p1 = figure(plot_width=800, plot_height=550, title="Deaths trajectory(LOG)",

             x_range=(0, 100), y_axis_type="log")

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days since 5 death daly increase'

p1.yaxis.axis_label = 'Deaths'

p1.xaxis.minor_tick_line_width = 10







p1.line(br_day[7:], moving_br, color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(br_day[-1], moving_br[-1], fill_color="white", size=5)



p1.line(eua_day[7:], moving_eua, color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(eua_day[-1], moving_eua[-1], fill_color="white", size=5)



p1.line(su_day[5:], moving_su, color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(su_day[-1], moving_su[-1], fill_color="white", size=5)





p1.line(it_day[6:], moving_it, color='#3EC358', 

        legend_label='Italy', line_width=1)

p1.circle(it_day[-1], moving_it[-1], fill_color="white", size=5)





p1.legend.location = "bottom_right"

output_notebook()

show(p1)



layout = Layout(

    paper_bgcolor='lightsteelblue',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Cases and deaths in brazil in " + today

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=index_brasil

           , y=casos_brasil),

    

    go.Bar(name='Deaths'

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

    title="Cases and deaths in US in " + today

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=index_eua

           , y=casos_eua),

    

    go.Bar(name='Deaths'

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

    title="Cases and deaths Italy in " + today

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=index_italia

           , y=casos_italia),

    

    go.Bar(name='Deaths'

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

    title="Cases and death in Sweden in " + today

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=index_suecia

           , y=casos_suecia),

    

    go.Bar(name='Deaths'

           , x=index_suecia

           , y=mortes_suecia

           , text= mortes_suecia

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_brasil, recuperados_brasil, marker='o',  color='green')

plt.plot(dias_brasil, mortes_brasil, marker='o',  color='red')





plt.title('COMPARISON OF RECOVERED VS DEATHS BRAZIL', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Recovered', 'Deads'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_italia, recuperados_italia, marker='o',  color='green')

plt.plot(dias_italia, mortes_italia, marker='o',  color='red')





plt.title('COMPARISON OF RECOVERED VS DEATHS ITALY', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Recovered', 'Deads'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_eua, recuperados_eua, marker='o',  color='green')

plt.plot(dias_eua, mortes_eua, marker='o',  color='red')







plt.title('COMPARISON OF RECOVERED VS DEATHS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Recovered', 'Deads'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(dias_suecia, recuperados_suecia, marker='o',  color='green')

plt.plot(dias_suecia, mortes_suecia, marker='o',  color='red')







plt.title('COMPARISON OF RECOVERED VS DEATHS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['Recovered', 'Deads'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

#separating the target (value to be forecasted) from the dates

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
# Start from Brazil
#Cases Brazil

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

plt.title('BRAZIL COMPARISON CONFIRMED VS FORECAST', size=30)

plt.xlabel('Days since 02/27/20', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['confirmed cases', 'forecast with Poly Regression'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



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

plt.title('BRAZIL COMPARISON CONFIRMED DEATHS VS FORECAST', size=30)

plt.xlabel('Days since 02/27/20', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['confirmed deaths', 'forecast with Poly Regression'],  prop={'size': 20}, loc="upper left")



plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

#US

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=4000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_eua, target_confirmado_eua)
pred_caso_eua = mlp.predict(futuro_eua)
plt.figure(figsize=(20, 12))

plt.plot(dias_eua, casos_eua, marker='x')

plt.plot(futuro_eua[0:len(dias_eua)],pred_caso_eua[0:len(casos_eua)], linestyle = 'dashed', color='purple')

plt.title('US COMPARISON CONFIRMED VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['confirmed cases', 'forecast with MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_mortes_eua, target_mortos_eua)
pred_mortes_eua = mlp.predict(futuro_eua)
plt.figure(figsize=(20, 12))

plt.plot(dias_eua, mortes_eua, marker = 'x')

plt.plot(futuro_eua[0:len(dias_eua)],pred_mortes_eua[0:len(mortes_eua)], linestyle = 'dashed', color='purple')

plt.title('US COMPARISON  CONFIRMED DEATHS VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['confirmed deaths', 'forecast with MLP'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Italy Predict

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=30000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_italia, target_confirmado_italia)
pred_caso_italia = mlp.predict(futuro_italia)
plt.figure(figsize=(20, 12))

plt.plot(dias_italia, casos_italia, marker = 'x')

plt.plot(futuro_italia[0:len(dias_italia)],pred_caso_italia[0:len(casos_italia)], linestyle = 'dashed', color='purple')

plt.title('Italy COMPARISON CONFIRMED VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['confirmed cases', 'forecast with SVM'],  prop={'size': 20}, loc="upper left")







plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

X_train, X_test, y_train, y_test = train_test_split(df_mortes_italia, target_mortos_italia, test_size=0.1, random_state=42)



mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=30000, shuffle=True)

print('fitting...')

mlp.fit(df_mortes_italia, target_mortos_italia)
pred_mortes_italia = mlp.predict(futuro_italia)
plt.figure(figsize=(20, 12))

plt.plot(dias_italia, mortes_italia, marker='x')

plt.plot(futuro_italia[0:len(mortes_italia)],pred_mortes_italia[0:len(mortes_italia)], linestyle = 'dashed', color='purple')

plt.title('Italy COMPARISON CONFIRMED DEATHS VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['confirmed deaths', 'forecast with MLP'],  prop={'size': 20}, loc="upper left")







plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

# Sweden predict

mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000,1000),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)

print('fitting...')

mlp.fit(df_confirmados_suecia, target_confirmado_suecia)
pred_caso_suecia = mlp.predict(futuro_suecia)
plt.figure(figsize=(20, 12))

plt.plot(dias_suecia, casos_suecia, marker='x')

plt.plot(futuro_suecia[0:len(dias_suecia)], pred_caso_suecia[0:len(casos_suecia)], linestyle = 'dashed', color='purple')

plt.title('SWEDEN COMPARISON CONFIRMED VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of cases', size = 30)

plt.legend(['confirmed cases', 'forecast with SVM'],  prop={'size': 20}, loc="upper left")







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

plt.title('SWEDEN COMPARISON CONFIRMED  DEATHS VS FORECAST', size=30)

plt.xlabel('Days since start of covid', size = 30)

plt.ylabel('Number of deaths', size = 30)

plt.legend(['confirmed deaths', 'forecast with MLP'],  prop={'size': 20}, loc="upper left")







plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

index = ['Predict', 'Real', 'Percentage difference', 'Gross difference']

pd.set_option('display.max_columns', 250)
columns =[i for i in datas_futuras_br]

forecast = [round(i, 0) for i in pred_caso_br]

difference =[int(p) - int(r) for p,r in zip(pred_caso_br, casos_brasil)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, casos_brasil) ]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_case_br = pd.DataFrame([forecast[40:], casos_brasil[40:], formated_percentage_difference[40:], difference[40:]],

                          columns=columns[40:], index = index)

difference = np.array(difference[40:])

percentage_difference = np.array(percentage_difference[40:])

print(f"The average gross error for confirmed cases in Brazil is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for confirmed cases in Brazil is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('CONFIRMED CASES FOR BRAZIL: ')

df_case_br
columns =[i for i in datas_futuras_br]

forecast = [round(i, 0)  for i in pred_mortes_br]

difference =[int(p) - int(r) for p,r in zip(pred_mortes_br, mortes_brasil)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, mortes_brasil)]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_death_br = pd.DataFrame([forecast[40:], mortes_brasil[40:], formated_percentage_difference[40:], 

                            difference[40:]], columns=columns[40:], index = index)

difference = np.array(difference[40:])

percentage_difference = np.array(percentage_difference[40:])

print(f"The average gross error for deaths in Brazil is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for deaths in Brazil is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('DEATHS IN BRASIL: ')

df_death_br
columns =[i for i in datas_futuras_us]

forecast = [round(i, 0) for i in pred_caso_eua]

difference =[int(p) - int(r) for p,r in zip(pred_caso_eua, casos_eua)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, casos_eua) ]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_case_us = pd.DataFrame([forecast[75:], casos_eua[75:], formated_percentage_difference[75:], difference[75:]],

                          columns=columns[75:], index = index)

difference = np.array(difference[75:])

percentage_difference = np.array(percentage_difference[75:])

print(f"The average gross error for confirmed cases in US is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for confirmed cases in US is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('CONFIRMED CASES IN US: ')

df_case_us
columns =[i for i in datas_futuras_us]

forecast = [round(i, 0)  for i in pred_mortes_eua]

difference =[int(p) - int(r) for p,r in zip(pred_mortes_eua, mortes_eua)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, mortes_eua)]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_death_us = pd.DataFrame([forecast[75:], mortes_eua[75:], formated_percentage_difference[75:], 

                            difference[75:]], columns=columns[75:], index = index)

difference = np.array(difference[75:])

percentage_difference = np.array(percentage_difference[75:])

print(f"The average gross error for deaths in US is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for deaths in US is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('DEATHS IN US ')

df_death_us
columns =[i for i in datas_futuras_it]

forecast = [round(i, 0) for i in pred_caso_italia]

difference =[int(p) - int(r) for p,r in zip(pred_caso_italia, casos_italia)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, casos_italia) ]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_case_it = pd.DataFrame([forecast[67:], casos_italia[67:], formated_percentage_difference[67:], difference[67:]],

                          columns=columns[67:], index = index)

difference = np.array(difference[67:])

percentage_difference = np.array(percentage_difference[67:])

print(f"The average gross error for confirmed cases in Italy is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for confirmed cases in Italy is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('CONFIRMED CASES IN ITALY : ')

df_case_it
columns =[i for i in datas_futuras_it]

forecast = [round(i, 0)  for i in pred_mortes_italia]

difference =[int(p) - int(r) for p,r in zip(pred_mortes_italia, mortes_italia)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, mortes_italia)]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_death_it = pd.DataFrame([forecast[67:], mortes_italia[67:], formated_percentage_difference[67:], 

                            difference[67:]], columns=columns[67:], index = index)

difference = np.array(difference[67:])

percentage_difference = np.array(percentage_difference[67:])

print(f"The average gross error for deaths in Italy is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for deaths in Italy is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('DEATHS IN ITALY: ')

df_death_it
columns =[i for i in datas_futuras_su]

forecast = [round(i, 0) for i in pred_caso_suecia]

difference =[int(p) - int(r) for p,r in zip(pred_caso_suecia, casos_suecia)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, casos_suecia) ]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_case_su = pd.DataFrame([forecast[66:], casos_suecia[66:], formated_percentage_difference[66:], difference[66:]],

                          columns=columns[66:], index = index)

difference = np.array(difference[66:])

percentage_difference = np.array(percentage_difference[66:])

print(f"The average gross error for confirmed cases in Sweden is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for confirmed cases in Sweden is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('CONFIRMED CASES IN SWEDEN: ')

df_case_su
columns =[i for i in datas_futuras_su]

forecast = [round(i, 0)  for i in pred_mortes_suecia]

difference =[int(p) - int(r) for p,r in zip(pred_mortes_suecia, mortes_suecia)]

percentage_difference = [(int(p) / int(r))*100 if r != 0 else 0 for p,r in zip(difference, mortes_suecia)]

percentage_difference = [round(m, 2) for m in percentage_difference]

formated_percentage_difference = [f'{i}%' for i in percentage_difference]

df_death_su = pd.DataFrame([forecast[66:], mortes_suecia[66:], formated_percentage_difference[66:], 

                            difference[66:]], columns=columns[66:], index = index)

difference = np.array(difference[66:])

percentage_difference = np.array(percentage_difference[66:])

print(f"The average gross error for deaths in Sweden is: {round(np.absolute(difference).mean(), 1 )}")

print(f'The average percentage error for deaths in Sweden is: {round(np.absolute(percentage_difference).mean(), 2)}%')
print('DEATHS IN SWEDEN: ')

df_death_su
plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, color='blue')

plt.plot(futuro_eua, pred_caso_eua, color='red')

plt.plot(futuro_italia, pred_caso_italia, color='green')

plt.plot(futuro_suecia, pred_caso_suecia, color='yellow')



plt.title('COMPARISON OF CASE FORECAST IN DIFFERENT COUNTRIES', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of cases', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

labels = ['US', 'Italy', 'Brazil', 'Sweden']

atualmente = [confirmados_eua,confirmados_italia,confirmados_brasil,confirmados_suecia]

apos_40 = [pred_caso_eua[-1], pred_caso_italia[-1], pred_caso_br[-1],  pred_caso_suecia[-1]]

apos_40 = [round(m, 0) for m in apos_40]

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='Current')

rects2 = ax.bar(x + width/2, apos_40, width, label=' forecast after 40 days')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Cases')

ax.set_title('Current cases vs forecast')

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
print('comparison of case forecast in the 4 countries')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2208057" data-url="https://flo.uri.sh/visualisation/2208057/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
eua_porc = round(((pred_caso_eua[-1] - confirmados_eua)/ confirmados_eua)*100 , 2)

br_porc =  round(((pred_caso_br[-1] - confirmados_brasil)/ confirmados_brasil)*100, 2)

it_porc =  round(((pred_caso_italia[-1] - confirmados_italia)/ confirmados_italia)*100, 2)

su_porc =  round(((pred_caso_suecia[-1] - confirmados_suecia)/ confirmados_suecia)*100, 2)





fig = plt.figure(figsize=(12, 7))

paises = ['US', 'Brazil', 'Italy', 'Sweden']

numeros = [eua_porc,br_porc,it_porc,su_porc]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('How many increased in 40 days', size = 20)

plt.title('% increase', size=20)

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

rects1 = ax.bar(x - width/2, atualmente, width, label='now')

rects2 = ax.bar(x + width/2, apos_40, width, label=' predict in day: ' + columns[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Cases per Million')

ax.set_title('Cases per million now vs predict ')

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
p1 = figure(plot_width=800, plot_height=550, title="Expected trajectory for covid-19 logarithmic",

             y_axis_type="linear", x_range=(80,190))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days'

p1.yaxis.axis_label = 'Progression of cases (logarithmic scale)'

p1.xaxis.minor_tick_line_width = 0





p1.line(np.squeeze(futuro_brasil)[40:], pred_caso_br[40:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(futuro_brasil[-1], pred_caso_br[-1], fill_color="white", size=5)



p1.line(np.squeeze(futuro_eua)[75:], pred_caso_eua[75:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(futuro_eua[-1], pred_caso_eua[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_suecia)[66:], pred_caso_suecia[66:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(futuro_suecia[-1], pred_caso_suecia[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_italia)[67:], pred_caso_italia[67:], color='#3EC358', 

        legend_label='Italy', line_width=1)

p1.circle(futuro_italia[-1], pred_caso_italia[-1], fill_color="white", size=5)







p1.legend.location = "bottom_right"

output_notebook()

show(p1)



plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, marker='o',  color='blue')

plt.plot(futuro_suecia, pred_caso_suecia, marker='o',  color='yellow')





plt.title('COMPARISON OF FORECAST BRAZIL VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of cases', size = 30)

plt.legend(['Brazil', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_caso_br, marker='o',  color='blue')

plt.plot(futuro_eua, pred_caso_eua, marker='o',  color='red')



plt.title('COMPARISON OF FORECAST BRAZIL VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of cases', size = 30)

plt.legend(['Brazil', 'US'],  prop={'size': 20}, loc="upper left")



plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_eua, pred_caso_eua, marker='o',  color='red')

plt.plot(futuro_italia, pred_caso_italia, marker='o',  color='green')





plt.title('COMPARISON OF FORECAST US VS ITALY', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of cases', size = 30)

plt.legend(['US', 'ITALY'],  prop={'size': 20}, loc="upper left")



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



plt.title('FORECAST  per million', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel(' cases  per million expected', size = 30)

plt.legend(['Brazil', 'US', 'Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, color='blue')

plt.plot(futuro_eua, pred_mortes_eua, color='red')

plt.plot(futuro_italia, pred_mortes_italia, color='green')

plt.plot(futuro_suecia, pred_mortes_suecia , color='yellow')



plt.title('COMPARISON OF DEATH FORECAST IN DIFFERENT COUNTRIES', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of Deaths', size = 30)

plt.legend(['Brazil', 'US','Italy', 'SWEDEN'],  prop={'size': 20}, loc="upper left")





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



paises = ['US', 'Italy', 'Brazil', 'Sweden']

numeros = [diferenca_eua_media, diferenca_italia_media, diferenca_brasil_media, diferenca_suecia_media]

numeros = [round(m, 0) for m in numeros]

fig = plt.figure(figsize=(12, 7))

rects = plt.bar(paises, numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)

ax = rects.patches



for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('DAILY INCREASE PREDICT', size = 20)

plt.title('daily increase of covid-19', size=20)

plt.show()
ax = plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil[76:], diferenca_brasil[35:], color='blue')

plt.plot(futuro_eua[76:], diferenca_eua, color='red')

plt.plot(futuro_italia[76:], diferenca_italia[8:], color='green')

plt.plot(futuro_suecia[76:], diferenca_suecia[9:], color='yellow')



plt.title('DAILY INCREASE PREDICT', size=30)

plt.xlabel('Days since begin predict', size = 30)

plt.ylabel('daily increase', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")

plt.xticks(size=15)



plt.yticks(size=15)

plt.show()
labels = ['US', 'Italy', 'Brazil', 'Sweden']

atualmente = [mortos_eua, mortos_italia, mortos_brasil, mortos_suecia]

apos_40 = [pred_mortes_eua[-1], pred_mortes_italia[-1], pred_mortes_br[-1],  pred_mortes_suecia[-1]]

apos_40 = [round(m, 0) for m in apos_40]

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='currently')

rects2 = ax.bar(x + width/2, apos_40, width, label=' forecast for ' + datas_futuras_br[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Deaths')

ax.set_title('currently deaths vs forecast')

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
print('death forecast 4 country')

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2288388" data-url="https://flo.uri.sh/visualisation/2288388/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
eua_porc = round(((pred_mortes_eua[-1] - mortos_eua)/ mortos_eua)*100 , 2)

br_porc =  round(((pred_mortes_br[-1] - mortos_brasil)/ mortos_brasil)*100, 2)

it_porc =  round(((pred_mortes_italia[-1] - mortos_italia)/ mortos_italia)*100, 2)

su_porc =  round(((pred_mortes_suecia[-1] - mortos_suecia)/ mortos_suecia)*100, 2)





fig = plt.figure(figsize=(12, 7))

paises = ['US', 'Brazil', 'Italy', 'Sweden']

numeros = [eua_porc,br_porc,it_porc,su_porc]

rects = plt.bar(paises,numeros, align='center', color=['red', 'green', 'blue', 'yellow'], width=0.5)



ax = rects.patches

for rect, label in zip(ax, numeros):

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, label,

            ha='center', va='bottom', fontsize=15)

    

plt.ylabel('% increase in deaths after 40 days', size = 20)

plt.title('% increase', size=20)

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

rects1 = ax.bar(x - width/2, atualmente, width, label='now')

rects2 = ax.bar(x + width/2, apos_40, width, label=' predict for' + columns[-1])



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Deaths per million')

ax.set_title('Deaths per million now vs predict ')

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
p1 = figure(plot_width=800, plot_height=550, title="Expected trajectory for logarithmic covid-19 deaths",

             y_axis_type="log", x_range=(75,190))

p1.grid.grid_line_alpha=1

p1.ygrid.band_fill_color = "#4682B4"

p1.ygrid.band_fill_alpha = 0.1

p1.xaxis.axis_label = 'Days'

p1.yaxis.axis_label = 'Deaths progression(logarithmic scale)'

p1.xaxis.minor_tick_line_width = 0





p1.line(np.squeeze(futuro_brasil)[40:], pred_mortes_br[40:], color='#3E4CC3', 

        legend_label='Brazil', line_width=1)

p1.circle(futuro_brasil[-1], pred_mortes_br[-1], fill_color="white", size=5)



p1.line(np.squeeze(futuro_eua)[75:], pred_mortes_eua[75:], color='#F54138', 

        legend_label='United States', line_width=1)

p1.circle(futuro_eua[-1], pred_mortes_eua[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_suecia)[66:], pred_mortes_suecia[66:], color='#DBAE23', 

        legend_label='Sweden', line_width=1)

p1.circle(futuro_suecia[-1], pred_mortes_suecia[-1], fill_color="white", size=5)





p1.line(np.squeeze(futuro_italia)[67:], pred_mortes_italia[67:], color='#3EC358', 

        legend_label='Italy', line_width=1)

p1.circle(futuro_italia[-1], pred_mortes_italia[-1], fill_color="white", size=5)







p1.legend.location = "bottom_right"

output_notebook()

show(p1)



fig = plt.figure(figsize=(12, 7))

labels = ['US', 'Italy', 'Brazil', 'Sweden']

atualmente = [(mortos_eua/confirmados_eua)*100, (mortos_italia/confirmados_italia) *100, 

           (mortos_brasil/confirmados_brasil)*100, (mortos_suecia/confirmados_suecia)*100]

atualmente = [round(num, 2) for num in atualmente]

apos_40 = [(pred_mortes_eua[-1]/pred_caso_eua[-1])*100, (pred_mortes_italia[-1]/pred_caso_italia[-1])*100,

           (pred_mortes_br[-1]/ pred_caso_br[-1])*100, (pred_mortes_suecia[-1]/pred_caso_suecia[-1])*100]

apos_40 = [round(m, 0) for m in apos_40]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width/2, atualmente, width, label='currently')

rects2 = ax.bar(x + width/2, apos_40, width, label=' forecast after 40 days')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('% mortality')

ax.set_title('Current vs predicted mortality in%')

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

#plt.plot(futuro_suecia, mort_su, color='yellow')



plt.title('Evolution of the mortality rate  Brazil vs USA', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Mortality rate', size = 30)

plt.legend(['Brasil', 'USA'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, marker='o',  color='blue')

plt.plot(futuro_suecia, pred_mortes_suecia, marker='o',  color='yellow')



plt.title('COMPARISON OF DEATH FORECAST BRAZIL VS SWEDEN', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of Deaths', size = 30)

plt.legend(['Brazil', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_italia, pred_mortes_italia, marker='o',  color='green')

plt.plot(futuro_eua, pred_mortes_eua, marker='o',  color='red')



plt.title('COMPARISON OF DEATH FORECAST ITALY VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of Deaths', size = 30)

plt.legend(['Italy', 'US'],  prop={'size': 20}, loc="upper left")







plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, pred_mortes_br, marker='o',  color='blue')

plt.plot(futuro_eua, pred_mortes_eua, marker='o',  color='red')



plt.title('COMPARISON OF DEATH FORECAST BRAZIL VS US', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('Expected number of Deaths', size = 30)

plt.legend(['Brazil', 'US'],  prop={'size': 20}, loc="upper left")







plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

eua_porc = []

br_porc = []

it_porc = []

su_porc = []



for i in pred_mortes_eua:

  porc = (i / 333545145)*100

  eua_porc.append(porc)

for i in pred_mortes_br:

  porc = (i / 217089238)*100

  br_porc.append(porc)

for i in pred_mortes_italia:

  porc = (i / 60017348)*100

  it_porc.append(porc)

for i in pred_mortes_suecia:

  porc = (i / 10174790)*100

  su_porc.append(porc)



plt.figure(figsize=(20, 9))

plt.plot(futuro_brasil, br_porc, color='blue')

plt.plot(futuro_eua, eua_porc, color='red')

plt.plot(futuro_italia, it_porc, color='green')

plt.plot(futuro_suecia, su_porc, color='yellow')



plt.title('PROGRESSION OF DEATHS IN RELATION TO POPULATION', size=30)

plt.xlabel('Days since covid started', size = 30)

plt.ylabel('% of deaths in relation to population', size = 30)

plt.legend(['Brazil', 'US','Italy', 'Sweden'],  prop={'size': 20}, loc="upper left")





plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
