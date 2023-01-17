# This Python 3 environment comes with many helpful analytics libraries installed

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Sorting and analysis
import pandas as pd
import numpy as np
import datetime as dt

# Visulization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium

# Converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#html embedding
from IPython.core.display import HTML

# hide Warnings
import warnings
warnings.filterwarnings('ignore')
file_mex = '../input/Mex_covid_update.csv'
mex_cov = pd.read_csv(file_mex)
mex_cov.head()
mex_cov.shape
mex_cov.info()
# Change the type to columns
mex_cov.iloc[:,[0,9,10,11]] = mex_cov.iloc[:,[0,9,10,11]].apply(pd.to_datetime) 

#Check for NaN values (exclude FECHA_DEF)
mex_cov.drop('FECHA_DEF', axis = 1).isna().sum()
# Confirmed COVID
covidc_table = mex_cov[mex_cov['RESULTADO'] == 'POSITIVO']
print("The number of people confirmed with COVID so far is: {:.0f}".format(covidc_table.shape[0]))

# COVID negative
covidn_table = mex_cov[mex_cov['RESULTADO'] == 'NEGATIVO']
print("The number of people with negative test against COVID so far is: {:.0f}".format(covidn_table.shape[0]))

# COVID standby
covidp_table = mex_cov[mex_cov['RESULTADO'] == 'PENDIENTE']
print("The number of people wainting for a result with COVID so far is: {:.0f}".format(covidp_table.shape[0]))

# Deaths
deaths = mex_cov[(~mex_cov['FECHA_DEF'].isna()) & (mex_cov['RESULTADO'] == 'POSITIVO')]
print("Number of deaths by COVID: {:.0f}".format(deaths.shape[0]))

# Total 
print("Total number of people analiyzed so far: {:.0f}".format(mex_cov.shape[0]))
# Deaths by date
mex_deaths = mex_cov[~mex_cov['FECHA_DEF'].isna()]
mex_deaths = mex_deaths.groupby('FECHA_DEF').size().reset_index()
mex_deaths.columns = ['FECHA_DEF', 'DEFUNCIONES']
cov_result = mex_cov.loc[:,['FECHA_SINTOMAS', 'RESULTADO']].pivot_table(index= 'FECHA_SINTOMAS', columns = 'RESULTADO', aggfunc = len, fill_value = 0).reset_index()
mex_cov[(mex_cov['FECHA_SINTOMAS'] <= '2020-02-01') & (mex_cov['RESULTADO'] != 'NEGATIVO')]
temp =  cov_result.melt(id_vars = 'FECHA_SINTOMAS', 
                       value_vars = ['NEGATIVO', 'PENDIENTE', 'POSITIVO'])

### bar plot
fig = px.bar(temp, x = 'FECHA_SINTOMAS', y = 'value', color = 'RESULTADO', title = 'Casos totales en México',
            color_discrete_sequence = ['#393e46', '#21bf73', '#fe9801'])
fig.update_layout(barmode = 'stack')
fig.show()
# Confirmed grouped by State
cov_sta_g = covidc_table.groupby('ENTIDAD_RES').apply(len).reset_index()
cov_sta_g.columns = ['ENTIDAD', 'TOTAL']

# Confirmed grouped by Date
cov_date_g = covidc_table.groupby('FECHA_SINTOMAS').apply(len).reset_index()
cov_date_g.columns = ['FECHA', 'TOTAL']
temp = cov_sta_g.sort_values(by = 'TOTAL', ascending= False)
temp = temp.reset_index(drop = True)
temp.style.background_gradient(cmap = 'Reds')
now = dt.datetime.now()
now = str(now.strftime("%d/%m/%Y"))
# By state
fig = px.bar(temp.sort_values('TOTAL', ascending = True), x = 'TOTAL', y = 'ENTIDAD', title = 'Confirmed cases in each State 17/06/2020'.format(now),
             text = 'TOTAL', orientation = 'h', range_x = [0, max(temp['TOTAL'] +150)])
fig.update_traces(marker_color ="#084177", opacity = 0.8, textposition = 'outside')
fig.show()
cov_date_g['TOTAL_ACCU'] = cov_date_g[['TOTAL']].cumsum()
temp = cov_date_g.sort_values(by = 'FECHA', ascending= True)
temp = temp.reset_index(drop = True)
temp.style.background_gradient(cmap = 'Greens')
# Confirmed and deaths grouped by Date
temp_d = covidc_table[~covidc_table['FECHA_DEF'].isna()]
temp_d = temp_d.groupby('FECHA_DEF').size().reset_index()
temp_d.columns = ['FECHA_DEF', 'DEFUNCIONES']
temp_d['DEF_ACU'] = temp_d['DEFUNCIONES'].cumsum()
# Merge both tables
temp_both = pd.merge(temp, temp_d, how= 'left', right_on = 'FECHA_DEF', left_on ='FECHA')
temp_both.iloc[:,[4,5]] = temp_both.iloc[:,[4,5]].fillna(0)
temp_both.iloc[:,[4,5]] = temp_both.iloc[:,[4,5]].astype('int')
temp_both.drop('FECHA_DEF', inplace = True, axis = 1)
temp_both
mex_cov[mex_cov['FECHA_DEF'] < mex_cov['FECHA_SINTOMAS']].iloc[:,:14]
temp_melt = temp_both.iloc[:,[0,2,4]].melt(id_vars = 'FECHA', value_vars = ['DEF_ACU', 'TOTAL_ACCU'],
                                     var_name = 'CASE', value_name = 'COUNT')
temp_melt['CASE'] = temp_melt['CASE'].replace({'DEF_ACU':'DEATHS',
                                         'TOTAL_ACCU':'CONFIRMED'})

# Area plot
fig = px.area(temp_melt,
             x = 'FECHA',
             y = 'COUNT',
             color = 'CASE',
             title = 'Cases in Mexico over time',
             color_discrete_sequence = ['red', 'grey'])
fig.show()
# Format the date
temp['FECHA'] = pd.to_datetime(temp['FECHA'])
temp['FECHA'] = temp['FECHA'].dt.strftime('%m/%d/%Y')

fig, ax1 = plt.subplots(figsize = (20,12))
# bar plot creation
ax1.set_title(label = 'Number of cases confirmed {}'.format(now), fontsize = 14)
ax1 = sns.barplot(x = 'FECHA', y='TOTAL', data = temp, palette = 'summer')
ax1.set_xlabel('Fecha', fontsize= 12)
ax1.set_ylabel('Confirmed cases')
ax1.tick_params(axis = 'y')
ax1.set_ylim(0,temp['TOTAL'].max() +10000)
# Specify we want to share the same x-axis
ax2 = ax1.twinx()
# line plot
ax2 = sns.lineplot(x = 'FECHA', y = 'TOTAL_ACCU', data = temp, color ='red')
ax2.set_ylabel('Total', fontsize = 12)
ax2.tick_params(axis = 'y', color = 'red')
ax2.set_ylim(0,temp['TOTAL_ACCU'].max() + 500)

# Rotate ticklabels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45, ha = 'right', fontsize = 7)
plt.tight_layout()
temp =  temp_both.melt(id_vars = 'FECHA', 
                       value_vars = ['TOTAL', 'DEFUNCIONES'])
temp['variable'] = temp['variable'].replace({''})

# New cases by date
fig = px.bar(temp, 
             x = 'FECHA', 
             y = 'value',
             color = 'variable',
             title = 'Casos por día',
             width = 1100, height = 600, range_y = [0, max(temp['value'] +500)])

#'#393e46', '#21bf73', '#fe9801'])
fig.update_layout(barmode = 'stack')
fig.show()
temp = covidc_table.groupby(['TIPO_PACIENTE', 'SEXO']).size().reset_index()
temp.rename(columns ={0:'CONTEOS'}, inplace = True)
fig = px.bar(temp, x="TIPO_PACIENTE", y= 'CONTEOS', color='SEXO', barmode='group',
             height=400)
fig.show()
temp = covidc_table[covidc_table['TIPO_PACIENTE'] == 'HOSPITALIZADO']

temp = temp.groupby(['INTUBADO', 'SEXO']).size().reset_index()
temp.rename(columns ={0:'CONTEOS'}, inplace = True)

fig = px.bar(temp, x="INTUBADO", y= 'CONTEOS', color='SEXO', barmode='group',
             height=400, title = 'Personas hospitalizadas y su condición')
fig.show()
temp = covidc_table[covidc_table['TIPO_PACIENTE'] == 'HOSPITALIZADO']
fig = px.box(temp, 
            x = 'INTUBADO',
            y = 'EDAD',
            color = 'SEXO',
            title = "Age of Confirmed Cases in Hospital"
            )
fig.show()
# Make an age range
labels =['0 a 10',
        '11 a 20',
        '21 a 30',
        '31 a 40',
        '41 a 50',
        '51 a 60',
        '61 a 70',
        '71 a 80',
        '> 81']
covidc_table['EDAD_RANGO'] = pd.cut(x = covidc_table['EDAD'], bins = [-1,10,20,30,40,50,60,70,80,covidc_table['EDAD'].max()], labels=labels)

# Select only the Intubated
temp = covidc_table[covidc_table['INTUBADO'] == 'SI']
temp = temp.groupby(['EDAD_RANGO', 'SEXO']).size().reset_index()
temp.rename(columns ={0:'CONTEOS'}, inplace = True)

#
fig = px.bar(temp, x="EDAD_RANGO", y= 'CONTEOS', color='SEXO', barmode='group',
             height=400, title = 'Rangos de edad de las personas hospitalizadas por COVID19')
fig.show()

# Following the intubated make a Pivot table
temp = pd.pivot_table(covidc_table, index='INTUBADO', columns = ['OBESIDAD'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp = temp.melt(id_vars = 'INTUBADO', value_vars = ['NO', 'SE IGNORA', 'SI'], var_name = 'OBESIDAD', value_name = 'CONTEO')

### bar plot
fig = px.bar(temp, x = 'INTUBADO', y = 'CONTEO', color = 'OBESIDAD', title = 'Condicion de Obesidad de las personas confirmadas con COVID19')
fig.update_layout(barmode = 'stack')
fig.show()
# Make subplots by Disease that can be affected in the COVID infection.
temp = covidc_table[covidc_table['TIPO_PACIENTE'] == 'HOSPITALIZADO']
#Obesity
temp_c = pd.pivot_table(temp, index='INTUBADO', columns = ['OBESIDAD'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp_c = temp_c.melt(id_vars = 'INTUBADO', value_vars = ['NO', 'SE IGNORA', 'SI'], var_name = 'OBESIDAD', value_name = 'CONTEO')
fig = px.bar(temp_c, x = 'INTUBADO', y = 'CONTEO', color = 'OBESIDAD', title = 'Condicion de Obesidad de las personas hospitalizadas por COVID19', barmode ='stack')
fig.show()

#Diabetes
temp_c = pd.pivot_table(temp, index='INTUBADO', columns = ['DIABETES'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp_c = temp_c.melt(id_vars = 'INTUBADO', value_vars = ['NO', 'SE IGNORA', 'SI'], var_name = 'DIABETES', value_name = 'CONTEO')
fig = px.bar(temp_c, x = 'INTUBADO', y = 'CONTEO', color = 'DIABETES', title = 'Condicion de Diabetes de las personas hospitalizadas por COVID19', barmode = 'stack')
fig.show()

#Hipertension
temp_c = pd.pivot_table(temp, index='INTUBADO', columns = ['HIPERTENSION'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp_c = temp_c.melt(id_vars = 'INTUBADO', value_vars = ['NO', 'SE IGNORA', 'SI'], var_name = 'HIPERTENSION', value_name = 'CONTEO')
fig = px.bar(temp_c, x = 'INTUBADO', y = 'CONTEO', color = 'HIPERTENSION', title = 'Condicion de Hipertension de las personas hospitalizadas por COVID19', barmode = 'stack')
fig.show()
temp_c = pd.pivot_table(covidc_table[covidc_table['TIPO_PACIENTE'] == "HOSPITALIZADO"], index='INTUBADO', columns = ['SECTOR'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp_c = temp_c.melt(id_vars = 'INTUBADO', var_name= 'SECTOR')

fig = px.bar(temp_c, 
            x = 'SECTOR',
            y = 'value',
            color = 'INTUBADO',
            title = 'Número de personas registradas como hospitalizadas y confirmadas con COVID en los distintos hospitales.',
            barmode = 'stack')
fig.show()
# Following the same interest, we explore the risk for the people with COVID if they have some disease.
temp = covidc_table
temp['COUNT_ENF'] = temp.iloc[:,[18,22,25]].eq('SI').sum(axis =1)
dic = {0:'No comorbilidades',
      1:'Una comorbilidad',
      2:'Dos comorbilidades',
      3:'Tres comorbilidades'}
temp['COUNT_ENF'] = temp['COUNT_ENF'].map(dic)


temp_c = pd.pivot_table(temp, index='COUNT_ENF', columns = ['INTUBADO'],
              aggfunc = 'size',
              fill_value = 0).reset_index()
temp_c = temp_c.melt(id_vars = 'COUNT_ENF', var_name = 'INTUBADO')

fig = px.bar(temp_c, 
            x = 'COUNT_ENF',
            y = 'value',
            color = 'INTUBADO',
            title = 'Número de co-morbilidades de las personas confirmadas con COVID19',
            category_orders= {'COUNT_ENF': ['No comorbilidades','Una comorbilidad','Dos comorbilidades']})
fig.show()
#  read the population data frame
popu = pd.read_csv("../input/data-covid19/poblacion_2017.csv")
popu['Estado'] = popu['Estado'].str.upper()
popu['Poblacion'] = popu['Poblacion'].str.replace(" ", "").astype('int')
popu.head()
# Confirmed deaths grouped by State
cov_de_g = covidc_table[~covidc_table['FECHA_DEF'].isna()].groupby(['ENTIDAD_RES']).apply(len).reset_index()
cov_de_g.columns = ['ENTIDAD', 'TOTAL']

temp = pd.merge(cov_de_g, popu, left_on="ENTIDAD", right_on='Estado').drop("Estado", axis = 1)
temp['rate'] = (temp['TOTAL']/temp['Poblacion']) * 10000
temp = temp.sort_values('rate', ascending = False)

# Making a plot
fig = go.Figure()

fig = fig.add_trace(go.Scatter(x=temp['ENTIDAD'], y=temp['rate'],
                    mode='lines+markers',
                    name='lines+markers'))
fig = fig.update_layout(showlegend=False, 
                  title = "Número de fallecimientos por COVID por cada 10000 habitantes", 
                 yaxis_title = "")

fig.show()
# Confirmed deaths grouped by State
cov_de_g = covidc_table[~covidc_table['FECHA_DEF'].isna()].groupby(['ENTIDAD_RES']).apply(len).reset_index()
cov_de_g.columns = ['ENTIDAD', 'TOTAL-D']

temp = pd.merge(cov_de_g, cov_sta_g, on="ENTIDAD")
temp['Letality'] = (temp['TOTAL-D']/temp['TOTAL'])*100 
temp = temp.sort_values("Letality", ascending = False)

# Making a plot
fig = go.Figure()

fig = fig.add_trace(go.Scatter(x=temp['ENTIDAD'], y=temp['Letality'],
                    mode='lines+markers',
                    name='lines+markers'))
fig = fig.update_layout(showlegend=False, 
                  title = "Fallecimientos por casos confirmados de COVID", 
                 yaxis_title = "% Falleciminetos por casos confirmados"
)

fig.show()
# How many COVID confirmed talk a ethnic languages? 1 means a positive value and 2 means a negative value.
covidc_table['HABLA_LENGUA_INDIG'].value_counts()
# How many COVID confirmed are migrants?
covidc_table['MIGRANTE'].value_counts()
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977789" data-url="https://flo.uri.sh/visualisation/1977789/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
mex_state_death = mex_cov[~mex_cov['FECHA_DEF'].isna()]
mex_state_death = mex_state_death.groupby('ENTIDAD_RES').size().reset_index().rename(columns = {0:'TOTAL_DEF',
                                                                                               'ENTIDAD_RES': 'ENTIDAD'})

# Merge
temp = pd.merge(mex_state_death, cov_sta_g,  on = 'ENTIDAD')

# Scatter plot Deaths vs Confiermed
fig = px.scatter(temp,
                x = 'TOTAL',
                y = 'TOTAL_DEF',
                color = 'ENTIDAD',
                text = 'ENTIDAD',
                title = 'Fallecimientos vs Confirmados por Estado.',
                labels = {'TOTAL_DEF':'Defunciones', 'TOTAL': 'Confirmados'})
fig.update_traces(textposition = 'top center')
fig.show()
cov_stamun_g = covidc_table.groupby(['ENTIDAD_RES', 'MUNICIPIO']).apply(len).reset_index()
cov_stamun_g.columns = ['ENTIDAD', 'MUNICIPIO', 'TOTAL']
#cov_stamun_g.head()

mex_state_death = mex_cov[~mex_cov['FECHA_DEF'].isna()]
mex_state_death = mex_state_death.groupby(['ENTIDAD_RES', 'MUNICIPIO']).size().reset_index().rename(columns = {0:'TOTAL_DEF',
                                                                                             'ENTIDAD_RES': 'ENTIDAD'})
#mex_state_death.head()
# Merge
temp = pd.merge(mex_state_death , cov_stamun_g,  on = ['ENTIDAD','MUNICIPIO'])
#temp.head()

fig = px.treemap(temp,
                path = ['ENTIDAD','MUNICIPIO'],
                values = 'TOTAL', height= 700,
                title = 'Numero de casos confirmados por Estado',
                color_discrete_sequence=px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()

fig = px.treemap(temp,
                path = ['ENTIDAD', 'MUNICIPIO'],
                values = 'TOTAL_DEF', height= 700,
                title = 'Numero de fallecimientos por COVID por Estado',
                color_discrete_sequence=px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()
first_date = covidc_table.groupby('ENTIDAD_RES')['FECHA_SINTOMAS'].agg(['min']).reset_index()
from datetime import timedelta
import random

# Last date
# ---------

test = covidc_table.groupby(['ENTIDAD_RES', 'FECHA_SINTOMAS']).size().reset_index().rename(columns = {0:'Confirmados'})
test = test.groupby(['ENTIDAD_RES', 'FECHA_SINTOMAS'])['Confirmados']
last_day = test.sum().diff().reset_index()

mask = last_day['ENTIDAD_RES'] != last_day['ENTIDAD_RES'].shift(1)
last_day.loc[mask, 'Confirmados'] = np.nan

last_day = last_day.groupby('ENTIDAD_RES')['FECHA_SINTOMAS'].agg(['max']).reset_index()

# First_last
# -------
first_last = pd.concat([first_date, last_day[['max']]], axis = 1)

# Added 1 more day, which will show the next day as the day on which last case appeared
first_last['max'] = first_last['max'] + timedelta(days = 1)

# No of days
first_last['Days'] = first_last['max'] - first_last['min']

# Task column  as State
first_last['Task'] = first_last['ENTIDAD_RES']

# rename
first_last.columns = ['ENTIDAD_RES', 'Start', 'Finish', 'Days', 'Task']
# sort by no. of days
first_last = first_last.sort_values('Days')

#first_last.head()

# Visualization
# ------------
# Produce random colors
clr = ["#"+''.join([random.choice('0123456789ABC') for j in range(6)]) for i in range(len(first_last))]

# Plot 
fig = ff.create_gantt(first_last,
                     index_col = 'ENTIDAD_RES',
                     colors = clr,
                     show_colorbar = False,
                     bar_width = 0.2,
                     showgrid_x = True,
                     showgrid_y = True,
                     height= 1600,
                     title = 'Duracion Epidemica')
fig.show()
temp = covidc_table[covidc_table['FECHA_DEF'] <= covidc_table['FECHA_SINTOMAS']]
temp.shape
temp.iloc[:,0:14]
covidc_table.groupby('NEUMONIA')['INTUBADO'].value_counts()
covidn_table.groupby('NEUMONIA')['INTUBADO'].value_counts()
# Get first and last datetime for final week of data
range_max = covidc_table['FECHA_SINTOMAS'].max()
range_min = range_max - dt.timedelta(days = 14)

table_conf_14 = covidc_table[(covidc_table['FECHA_SINTOMAS'] >= range_min) & ((covidc_table['FECHA_SINTOMAS'] <= range_max))]
table_conf_14.shape
temp = table_conf_14.groupby(["ENTIDAD_RES", "MUNICIPIO"]).apply(len).reset_index()
temp.columns = ["ENTIDAD", "MUNICIPIO", "TOTAL"]
#temp.head()

# Tree plot
fig = px.treemap(temp,
                path = ["ENTIDAD", "MUNICIPIO"],
                values = "TOTAL", height= 700,
                title = "Numero de casos confirmados en los ultimos 14 días",
                color_discrete_sequence=px.colors.qualitative.Prism)
fig.data[0].textinfo = "label+text+value"
fig.show()