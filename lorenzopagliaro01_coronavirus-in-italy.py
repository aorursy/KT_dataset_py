# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from datetime import datetime

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px
data_national = pd.read_csv("/kaggle/input/coronavirus-italian-data/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
#add daily cases/deaths/recovered columns

data_national['new_cases'] = data_national['totale_casi'].diff()

data_national['new_deaths'] = data_national['deceduti'].diff()

data_national['new_recovered'] = data_national['dimessi_guariti'].diff()

data_national['new_swabs'] = data_national['tamponi'].diff()

data_national['new_unique_tested'] = data_national['casi_testati'].diff()

data_national['new_terapia_intensiva'] = data_national['terapia_intensiva'].diff()

#add a day/day-1 percentage change for new_cases

data_national['daily_cases_perc_change'] = round((data_national['new_cases'].pct_change(1))*100,2)

data_national['daily_swab_perc_change'] = round((data_national['new_swabs'].pct_change(1))*100,2)

data_national['daily_unique_tested_perc_change'] = round((data_national['new_unique_tested'].pct_change(1))*100,2)

#detect ratio

data_national['detect_ratio_swabs'] = round((data_national['new_cases'] / data_national['new_swabs'])*100,2)

data_national['detect_ratio_cases'] = round((data_national['new_cases'] / data_national['new_unique_tested'])*100,2)

data_national.tail(10)
data_region = pd.read_csv("/kaggle/input/coronavirus-italian-data/dati-regioni/dpc-covid19-ita-regioni.csv")

data_region.tail(30)
#regional data preparation



data_region_Abruzzo = data_region[(data_region['denominazione_regione'] == 'Abruzzo')]

data_region_Basilicata = data_region[(data_region['denominazione_regione'] == 'Basilicata')]

data_region_Bolzano = data_region[(data_region['denominazione_regione'] == 'P.A. Bolzano')]

data_region_Calabria = data_region[(data_region['denominazione_regione'] == 'Calabria')]

data_region_Campania = data_region[(data_region['denominazione_regione'] == 'Campania')]

data_region_EmiliaR = data_region[(data_region['denominazione_regione'] == 'Emilia-Romagna')]

data_region_Friuli = data_region[(data_region['denominazione_regione'] == 'Friuli Venezia Giulia')]

data_region_Lazio = data_region[(data_region['denominazione_regione'] == 'Lazio')]

data_region_Liguria = data_region[(data_region['denominazione_regione'] == 'Liguria')]

data_region_Lombardia = data_region[(data_region['denominazione_regione'] == 'Lombardia')]

data_region_Marche = data_region[(data_region['denominazione_regione'] == 'Marche')]

data_region_Molise = data_region[(data_region['denominazione_regione'] == 'Molise')]

data_region_Piemonte = data_region[(data_region['denominazione_regione'] == 'Piemonte')]

data_region_Puglia = data_region[(data_region['denominazione_regione'] == 'Puglia')]

data_region_Sardegna = data_region[(data_region['denominazione_regione'] == 'Sardegna')]

data_region_Sicilia = data_region[(data_region['denominazione_regione'] == 'Sicilia')]

data_region_Toscana = data_region[(data_region['denominazione_regione'] == 'Toscana')]

data_region_Trento = data_region[(data_region['denominazione_regione'] == 'P.A. Trento')]

data_region_Umbria = data_region[(data_region['denominazione_regione'] == 'Umbria')]

data_region_VAosta = data_region[(data_region['denominazione_regione'] == "Valle d'Aosta")]

data_region_Veneto = data_region[(data_region['denominazione_regione'] == 'Veneto')]



def region_apply(region):

    for x in region:

        x['new_cases'] =  x['totale_casi'].diff()

        x['new_deaths'] = x['deceduti'].diff()

        x['new_recovered'] = x['dimessi_guariti'].diff()

        x['new_swabs'] = x['tamponi'].diff()

        #add a day/day-1 percentage change for new_cases

        x['daily_cases_perc_change'] = round((x['new_cases'].pct_change(1))*100,2)

        x['daily_swab_perc_change'] = round((x['new_swabs'].pct_change(1))*100,2)

        #detect ratio

        x['detect_ratio'] = round((x['new_cases'] / x['new_swabs'])*100,2)

        return; 



region_apply([data_region_Abruzzo])  

region_apply([data_region_Basilicata]) 

region_apply([data_region_Bolzano])

region_apply([data_region_Calabria])

region_apply([data_region_Campania])

region_apply([data_region_EmiliaR])

region_apply([data_region_Friuli])

region_apply([data_region_Lazio])

region_apply([data_region_Liguria])

region_apply([data_region_Lombardia])

region_apply([data_region_Marche])

region_apply([data_region_Molise])

region_apply([data_region_Piemonte])

region_apply([data_region_Puglia])

region_apply([data_region_Sardegna])

region_apply([data_region_Sicilia])

region_apply([data_region_Toscana])

region_apply([data_region_Trento])

region_apply([data_region_VAosta])

region_apply([data_region_Veneto])





data_region_Nordovest = data_region[(data_region.denominazione_regione.isin(['Piemonte', 'Lombardia', 'Liguria',"Valle d'Aosta"]))]

data_region_Nordest = data_region[(data_region.denominazione_regione.isin(['Emilia-Romagna', 'P.A. Bolzano', 'P.A. Trento', 'Veneto', 'Friuli Venezia Giulia']))]

data_region_Centro = data_region[(data_region.denominazione_regione.isin(['Toscana', 'Umbria', 'Marche', 'Lazio']))]

data_region_Sudisole = data_region[(data_region.denominazione_regione.isin(['Abruzzo', 'Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna']))]



cases_Nordovest = data_region_Nordovest.groupby('data').sum()

region_apply([cases_Nordovest])  

cases_Nordovest['data'] = cases_Nordovest.index



cases_Nordest = data_region_Nordest.groupby('data').sum()

region_apply([cases_Nordest])  

cases_Nordest['data'] = cases_Nordest.index



cases_Centro = data_region_Centro.groupby('data').sum()

region_apply([cases_Centro])  

cases_Centro['data'] = cases_Centro.index



cases_Sudisole = data_region_Sudisole.groupby('data').sum()

region_apply([cases_Sudisole])  

cases_Sudisole['data'] = cases_Sudisole.index



cases_Nordovest.tail(5)
fig2 = px.bar(data_national, x='data', y='totale_casi',

             hover_data=['totale_casi'], color='totale_casi',

             height=600, color_continuous_scale='Sunsetdark')



fig2.update_layout(title_text='Total COVID19 Cases - Italy',

                  xaxis_rangeslider_visible=True)

fig2.update_yaxes(tick0=0, dtick=25000,  gridcolor='White')

fig2.show()
fig22 = px.bar(data_national, x='data', y='totale_positivi',

             hover_data=['totale_positivi'], color='totale_positivi',

             height=600, color_continuous_scale='Sunsetdark')



fig22.update_layout(title_text='Active COVID19 Cases - Italy',

                  xaxis_rangeslider_visible=True)

fig22.update_yaxes(tick0=0, dtick=10000,  gridcolor='White')

fig22.show()
fig = go.Figure()



fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Nordovest['data'], y=cases_Nordovest['new_cases'], name="North-West",

                         line_color='red'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Nordest['data'], y=cases_Nordest['new_cases'], name="North-East",

                         line_color='green'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Centro['data'], y=cases_Centro['new_cases'], name="Center",

                         line_color='darkviolet'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Sudisole['data'], y=cases_Sudisole['new_cases'], name="South and Islands",

                         line_color='darkblue'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_cases'], name="All Italy",

                         line_color='deepskyblue'))



fig.update_layout(title_text='Daily Coronavirus new cases - All Italy and Regions',

                  xaxis_rangeslider_visible=True)





fig.show()
figti = px.bar(data_national, x='data', y='terapia_intensiva',

             hover_data=['terapia_intensiva'], color='terapia_intensiva',

             height=600, color_continuous_scale='jet')



figti.update_layout(title_text='Total COVID19 Intensive Care Patients - Italy, estimated cap is 7500',

                  xaxis_rangeslider_visible=True)

figti.update_yaxes(tick0=0, dtick=250,  gridcolor='White')



figti.show()
fig = go.Figure()





fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_deaths'], name="Daily Deaths",

                         line_color='red'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_terapia_intensiva'], name="Daily Intensive Care",

                         line_color='orange'))







fig.update_layout(title_text='Daily Coronavirus Deaths and Intensive Care Patients - Italy',

                  xaxis_rangeslider_visible=True)



fig.update_yaxes(tick0=0, dtick=100)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Nordovest['data'], y=cases_Nordovest['new_swabs'], name="North-West",

                         line_color='red'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Nordest['data'], y=cases_Nordest['new_swabs'], name="North-East",

                         line_color='green'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Centro['data'], y=cases_Centro['new_swabs'], name="Center",

                         line_color='darkviolet'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=cases_Sudisole['data'], y=cases_Sudisole['new_swabs'], name="South and Islands",

                         line_color='darkblue'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_swabs'], name="All Italy",

                         line_color='deepskyblue'))



fig.update_layout(title_text='Daily swabs - All Italy and Regions',

                  xaxis_rangeslider_visible=True)





fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_cases'], name="Daily Cases",

                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_swabs'], name="Daily swabs",

                         line_color='purple'))

fig.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['new_unique_tested'], name="Daily unique tested",

                         line_color='red'))

fig.update_layout(title_text='Daily Coronavirus new cases and swabs - Italy',

                  xaxis_rangeslider_visible=True)



fig.update_yaxes(tick0=0, dtick=10000)



fig.show()



fig3 = go.Figure()



fig3.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['detect_ratio_swabs'], name="Daily detect ratio - Italy",

                         line_color='purple'))

fig3.add_trace(go.Scatter(mode = "lines+markers", x=data_national['data'], y=data_national['detect_ratio_cases'], name="Daily unique detect ratio - Italy",

                         line_color='red'))





fig3.update_layout(title_text="Daily Swabs detect ratio - Italy",

                  xaxis_rangeslider_visible=True)

fig3.update_yaxes(dtick=5)