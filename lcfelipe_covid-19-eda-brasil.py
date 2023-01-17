import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly packages
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *

#widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#maps
import json
from pandas import json_normalize

import folium
from folium.plugins import MiniMap
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
from folium.map import *

#os
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dirname='/kaggle/input'
bpop = pd.read_csv(os.path.join(dirname,'corona-virus-brazil','brazil_population_2019.csv'),error_bad_lines=False)
bcovidMacro = pd.read_csv(os.path.join(dirname,'corona-virus-brazil','brazil_covid19_macro.csv'),error_bad_lines=False)
bcovid = pd.read_csv(os.path.join(dirname,'corona-virus-brazil','brazil_covid19.csv'),error_bad_lines=False)
bcities= pd.read_csv(os.path.join(dirname,'corona-virus-brazil','brazil_cities_coordinates.csv'),error_bad_lines=False)
bcitiesCovid = pd.read_csv(os.path.join(dirname,'corona-virus-brazil','brazil_covid19_cities.csv'),error_bad_lines=False)

bcovidMacro.fillna(0,inplace=True)

bcovidMacro['cases_log'] = np.log(bcovidMacro['cases'])
bcovidMacro['deaths_log'] = np.log(bcovidMacro['deaths'])
beds_supplies = pd.read_csv(os.path.join(dirname,'icu-beds-brazil','lista_insumos_e_leitos.csv'),delimiter=';',error_bad_lines=False)
print('Período da análise:' + min(bcovid['date']) + ' / ' + max(bcovid['date']))
ev = pd.DataFrame(bcovidMacro.cases.diff().fillna(0))
ev = ev.join(bcovidMacro['date'])

ev2 = pd.DataFrame(bcovidMacro.deaths.diff().fillna(0))
ev2 = ev2.join(bcovidMacro['date'])

ev3 = pd.DataFrame(bcovidMacro.recovered.diff().fillna(0))
ev3 = ev3.join(bcovidMacro['date'])

ev['cases_log'] = ev['cases'].apply(lambda x: np.log(x))

ev2['deaths_log'] = ev2['deaths'].apply(lambda x: np.log(x))
layout = Layout(
    title="Daily cases/deaths",
)

fig = go.Figure(data=[
    go.Scatter(name='Cases', x=ev.date, y=ev['cases']),
    go.Scatter(name='Deaths', x=ev2.date, y=ev2['deaths']),
    
])
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Qty')
fig.update_layout(barmode='stack')
fig['layout'].update(layout)

out = fig.show()
layout = Layout(
    title="Total",
)

fig = go.Figure(data=[
    go.Scatter(name='Cases', x=bcovidMacro.date, y=bcovidMacro['cases']),
    go.Scatter(name='Deaths', x=bcovidMacro.date, y=bcovidMacro['deaths']),
    go.Scatter(name='Recovered', x=bcovidMacro.date, y=bcovidMacro['recovered'])
    
])
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Qty')
fig.update_layout(barmode='stack')
fig['layout'].update(layout)

out = fig.show()
layout = Layout(
    title="LOG",
)

fig = go.Figure(data=[
    go.Scatter(name='Cases', x=bcovidMacro.date, y=round(bcovidMacro['cases_log'],2)),
    go.Scatter(name='Deaths', x=bcovidMacro.date, y=round(bcovidMacro['deaths_log'],2)),
    
])
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Qty')
fig.update_layout(barmode='stack')
fig['layout'].update(layout)

fig.show()
mun_d = bcitiesCovid[bcitiesCovid['deaths']>0].groupby('date').count()[['deaths']]
mun_c = bcitiesCovid[bcitiesCovid['cases']>0].groupby('date').count()[['cases']]
mun = pd.merge(mun_c,mun_d,left_index=True,right_index=True)

municipios_br = len(bcities)
mun['perc_total_d'] = mun['deaths'] / municipios_br*100
mun['perc_total_c'] = mun['cases'] / municipios_br*100
mun.reset_index(inplace=True)
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=mun.index, y=round(mun['perc_total_c'],2), name="% of cities with confirmed cases"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=mun.index, y=round(mun['perc_total_d'],2), name="% of cities with confirmed deaths",mode='markers'),
    secondary_y=False,
)
fig.update_layout(title='% of cities with cases/deaths',)

# Set x-axis title
fig.update_xaxes(title_text="Days passed")

# Set y-axes titles
fig.update_yaxes(title_text="<b>%</b> ", secondary_y=False)

fig.show()
evSemanalMacro=bcovidMacro.groupby('week').max()
evSemanalMacro= evSemanalMacro[evSemanalMacro.columns[2:]]

maxv=max(list(evSemanalMacro.index))
minv=min(list(evSemanalMacro.index))

w = widgets.IntSlider(
    value=maxv,
    min=int(min(list(evSemanalMacro.index))),
    max=int(max(list(evSemanalMacro.index))),
    description='weeks:'
)
def slider(val):
    xs = []
    for i in range(minv,val+1):
        xs.append(i)
        
    layout = Layout(
        title="Total",
    )

    fig = go.Figure(data=[
        go.Bar(name='Cases', x=xs, y=evSemanalMacro['cases']),
        go.Bar(name='Deaths', x=xs, y=evSemanalMacro['deaths'])

    ])
    fig.update_xaxes(title_text='weeks')
    fig.update_yaxes(title_text='Qt')
    fig.update_layout(barmode='stack')
    fig['layout'].update(layout)

    fig.show()
out = interact(slider,val=w)
df = bcovid.groupby(['date','state'])['deaths'].sum().reset_index()
table = pd.pivot_table(df, values='deaths', index=['date'],columns=['state'], aggfunc=np.sum)
s = table.loc['2020-03-29']
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,6), dpi=144)
colors = plt.cm.Dark2(range(6))
y = s.index
width = s.values
out = ax.barh(y=y, width=width, color=colors)
def nice_axes(ax):
    ax.set_facecolor('.8')
    ax.tick_params(labelsize=8, length=0)
    ax.grid(True, axis='x', color='white')
    ax.set_axisbelow(True)
    [spine.set_visible(False) for spine in ax.spines.values()]
    
nice_axes(ax)
fig
fig, ax_array = plt.subplots(nrows=1, ncols=3, figsize=(8, 6), dpi=144, tight_layout=True)
dates = ['2020-03-29', '2020-04-30', '2020-05-31']
for ax, date in zip(ax_array, dates):
    s = table.loc[date]
    y = table.loc[date].rank(method='first').values
    ax.barh(y=y, width=s.values, color=colors, tick_label=s.index)
    ax.set_title(date, fontsize='smaller')
    nice_axes(ax)
df2 = table.loc['2020-03-29':'2020-03-31']
df2
df2 = df2.reset_index()
df2
df2.index = df2.index * 5
df2
last_idx = df2.index[-1] + 1
df_expanded = df2.reindex(range(last_idx))
df_expanded
df_expanded['date'] = df_expanded['date'].fillna(method='ffill')
df_expanded = df_expanded.set_index('date')
df_expanded
df_rank_expanded = df_expanded.rank(axis=1, method='first')
df_rank_expanded
df_rank_expanded = df_rank_expanded.interpolate()
df_rank_expanded
df_expanded = df_expanded.interpolate()
df_expanded
fig, ax_array = plt.subplots(nrows=1, ncols=6, figsize=(12, 8), 
                             dpi=144, tight_layout=True)
labels = df_expanded.columns
for i, ax in enumerate(ax_array.flatten()):
    y = df_rank_expanded.iloc[i]
    width = df_expanded.iloc[i]
    ax.barh(y=y, width=width, color=colors, tick_label=labels)
    nice_axes(ax)
ax_array[0].set_title('2020-03-29')
ax_array[-1].set_title('2020-03-30')
def prepare_data(df, steps=7):
    df = df.reset_index()
    df.index = df.index * steps
    last_idx = df.index[-1] + 1
    df_expanded = df.reindex(range(last_idx))
    df_expanded['date'] = df_expanded['date'].fillna(method='ffill')
    df_expanded = df_expanded.set_index('date')
    df_rank_expanded = df_expanded.rank(axis=1, method='first')
    df_expanded = df_expanded.interpolate()
    df_rank_expanded = df_rank_expanded.interpolate()
    return df_expanded, df_rank_expanded

df_expanded, df_rank_expanded = prepare_data(table)
from matplotlib.animation import FuncAnimation

def init():
    ax.clear()
    nice_axes(ax)
    ax.set_ylim(.2, 6.8)

def update(i):
    for bar in ax.containers:
        bar.remove()
    y = df_rank_expanded.iloc[i]
    width = df_expanded.iloc[i]
    ax.barh(y=y, width=width, color=colors, tick_label=labels)
    date_str = df_expanded.index[i]
    ax.set_title(f'COVID-19 Deaths by state - {date_str}', fontsize='smaller')
    
fig = plt.Figure(figsize=(8, 5), dpi=144)
ax = fig.add_subplot()
anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_expanded), 
                     interval=100, repeat=False)
## REMOVED FOR PERFORMANCE
### TURN THE CELL BACK TO CODE TO SEE BAR CHART RACE
from IPython.display import HTML
html = anim.to_html5_video()
HTML(html)
def bar_chart(var):
    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=var+" by region & state",
    )

    df = bcovid[bcovid['date']==bcovid['date'].max()].sort_values(by=['region','cases'],ascending=False)

    fig = px.bar(df, y="region", x=var, color=var, orientation="h",
                 color_continuous_scale='Bluered', hover_name="state",)


    fig.update_xaxes(title_text='Qty')
    fig.update_yaxes(title_text='Region')
    fig['layout'].update(layout)

    fig.show()
out = interact(bar_chart,var=list(bcovid.columns[3:]))
layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title="Share by state",
)

fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Cases', 'Deaths'])

df = bcovid[bcovid['date']==bcovid['date'].max()].sort_values(by=['region','cases'],ascending=False)

values_cases = df.groupby('region').sum()[['cases']].reset_index()
values_deaths= df.groupby('region').sum()[['deaths']].reset_index()

fig.add_trace(go.Pie(labels=list(values_cases['region']), values=list(values_cases['cases']),
                     name="cases"), 1, 1)
fig.add_trace(go.Pie(labels=list(values_deaths['region']), values=list(values_deaths['deaths']),
                     name="Deaths"), 1, 2)
    


fig['layout'].update(layout)

fig.show()
#Preparação dos dados
df = bcovid[bcovid['date']==bcovid['date'].max()].sort_values(by=['region','cases'],ascending=False)

df['country']='Brazil'

bra = df.groupby(['country']).sum().reset_index()
reg = df.groupby(['country','region']).sum().reset_index()
sta = df.groupby(['country','region','state']).sum().reset_index()
#Casos

#Brazil
labels = list(bra['country'])
parent = ['']
values = list(bra['cases'])
#Regioes
labels+=list(reg['region'])
parent+=list(reg['country'])
values+=list(reg['cases'])
#Estados
labels+=list(sta['state'])
parent+=list(sta['region'])
values+=list(sta['cases'])

#Mortes
#Brazil
labels2 = list(bra['country'])
parent2 = ['']
values2 = list(bra['deaths'])
#Regioes
labels2+=list(reg['region'])
parent2+=list(reg['country'])
values2+=list(reg['deaths'])
#Estados
labels2+=list(sta['state'])
parent2+=list(sta['region'])
values2+=list(sta['deaths'])

#---------GRAFICO-------------#

fig = go.Figure()

fig.add_trace(go.Sunburst(
    labels=labels,
    parents=parent,
    values = values,
    domain=dict(column=0)
    ,branchvalues="total"
))

fig.add_trace(go.Sunburst(
    labels=labels2,
    parents=parent2,
    values = values2,
    domain=dict(column=1)
    ,branchvalues="total"
))

fig.update_layout(
    grid= dict(columns=2, rows=1),
   # margin = dict(t=0, l=0, r=0, b=0)
)

fig.show()
bcitiesCovid['code']  = bcitiesCovid['code'].apply(lambda x: int(x))
covid_pop = bcitiesCovid.merge(bpop.drop(['state','city'],axis=1), left_on='code', right_on='city_code')
covid_pop = covid_pop[covid_pop['date']==covid_pop['date'].max()]
covid_pop.drop(['code','city_code','health_region_code'],axis=1,inplace=True)
covid_pop.sort_values(by=['state_code'],inplace=True)
covid_pop.reset_index(drop=True,inplace=True)
covid_pop = covid_pop.groupby(['state','state_code']).sum()
covid_pop.reset_index(inplace=True)

covid_pop['letalidade'] = round((covid_pop['deaths'] / covid_pop['cases']) * 100,3)

covid_pop['casos100k'] = round((covid_pop['cases'] / covid_pop['population']) * 100000,3)
covid_pop['mortos100k'] = round((covid_pop['deaths'] / covid_pop['population']) * 100000,3)

covid_pop['date'] = max(bcovid['date'])
covid_pop['log_casos']=np.log(covid_pop['cases'])
covid_pop['log_mortos']=np.log(covid_pop['deaths'])
beds_supplies = pd.merge(beds_supplies[beds_supplies.columns[14:20]],beds_supplies[beds_supplies.columns[0]], left_index=True, right_index=True)

covid_pop = pd.merge(covid_pop,beds_supplies,left_on='state',right_on='uf')
covid_pop['uti100k'] = round((covid_pop['Leitos UTI adulto'] / covid_pop['population']) * 100000,3)
with open(os.path.join(dirname,'json-areas','estados.json'), 'r') as f:
    data = json.load(f)

geodata = json_normalize(data['features'])
#Gera arquivo com novas propriedades
z = 0

for feat in data['features']:
    id_ibge = int(data['features'][z]['properties']['codigo_ibg'])

    df = covid_pop[covid_pop['state_code'] == id_ibge].drop_duplicates()
    df.fillna(0,inplace=True)
    #print(df)
    if (len(df) > 0 & len(covid_pop) < 27):
        
        #mun = df['name'].iloc[0]
        atu = df['date'].iloc[0]
        casos = df['cases'].iloc[0]
        mortos = df['deaths'].iloc[0]
        pop = df['population'].iloc[0]
        let = df['letalidade'].iloc[0]
        c_cemk = df['casos100k'].iloc[0]
        m_cemk = df['mortos100k'].iloc[0]
        u_cemk = df['uti100k'].iloc[0]
        cod_ibge = df['state_code'].iloc[0]
        pop = df['population'].iloc[0]

        #data['features'][z]['properties']['name'] = mun
        
        data['features'][z]['properties']['codigo_ibg'] = int(cod_ibge)
        data['features'][z]['properties']['letalidade'] = "{:,.2f}".format(let)
        data['features'][z]['properties']['populacao'] = "{:,}".format(int(pop))
        data['features'][z]['properties']['casos'] = "{:,}".format(int(casos))
        data['features'][z]['properties']['mortos'] = "{:,}".format(int(mortos))
        data['features'][z]['properties']['casos100k'] = "{:,.2f}".format(c_cemk)
        data['features'][z]['properties']['mortos100k'] = "{:,.2f}".format(m_cemk)
        data['features'][z]['properties']['uti100k'] = "{:,.2f}".format(u_cemk)
        data['features'][z]['properties']['dta_atu'] = atu
           
    z =z+1
path_json = './mapa_estado.json'

new_features = []

for element in data["features"]:
    new_features.append(element)
        
data["features"] = new_features

with open(os.path.join(path_json), 'w') as f:
    json.dump(data, f,indent=4)
def plot_map(indicador,export):

    maps = folium.Map(location=[bcities['lat'].mean(),bcities['long'].mean()], zoom_start=5)
#layer 1
    folium.Choropleth(
        geo_data=path_json,
        data=covid_pop,
        columns=['state_code', indicador],
        key_on='feature.properties.codigo_ibg',
        fill_color='YlOrRd',
        fill_opacity=0.9,
        line_opacity=0.2,
        legend_name=indicador,
        highlight=True,
        nan_fill_color='grey',
        nan_fill_opacity=0.4,
        show = True,
        name=indicador,
        overlay=True
    ).add_to(maps)

#tooltips
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}
    
    tool = folium.features.GeoJson(
        path_json,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=['sigla','populacao','casos','mortos','letalidade','casos100k','mortos100k','uti100k','dta_atu'],
            aliases=['Estado: ','População: ','Casos: ','Mortos: ','Letalidade %: '
                         ,'Casos / 100k: ','Mortos / 100k: ', 'Uti / 100K','Data Atualização: '],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )

    maps.add_child(tool)
    maps.keep_in_front(tool)

    folium.LayerControl(autoZIndex=False, collapsed=True).add_to(maps)


    #--------------#export html#--------------#
    if export == True:
        maps.save('covid_mapa_'+indicador+'.html')

    return maps
list_ind = list(covid_pop.columns[2:])

remove_list = ['date','uf']
for ind in remove_list:
    list_ind.remove(ind)

out = interact(plot_map,indicador=list_ind,export=[False,True])
!pip install wget

import wget
url = 'https://secweb.procergs.com.br/isus-covid/api/v1/export/csv/hospitais'
filename = wget.download(url)
hospital_beds = pd.read_csv(filename, encoding='latin-1',delimiter=';')
hospital_beds['just_date'] = pd.to_datetime(hospital_beds['DATA INCLUSAO REGISTRO']).dt.date
last_date = hospital_beds['just_date'].max()

from dateutil.relativedelta import relativedelta

last_date2 = last_date + relativedelta(days = -1)
#adult icu in use
hospital_beds['uti_a_util'] = hospital_beds['NUMERO PACIENTES ADULTOS INTERNADOS EM LEITOS UTI  (SUS  PRIVADO)']/hospital_beds['NUMERO LEITOS UTI ADULTO (SUS  PRIVADO)']*100
hospital_beds['uti_a_util'].fillna(0,inplace=True)
hospital_beds.columns
rs = hospital_beds[hospital_beds['just_date']==last_date].groupby('SIGLA_UF').sum()
rs['uti_a_util'] = rs['NUMERO PACIENTES ADULTOS INTERNADOS EM LEITOS UTI  (SUS  PRIVADO)']/rs['NUMERO LEITOS UTI ADULTO (SUS  PRIVADO)']*100
rs['uti_a_util'].fillna(0,inplace=True)

rs2 = hospital_beds[hospital_beds['just_date']==last_date2].groupby('SIGLA_UF').sum()
rs2['uti_a_util'] = rs2['NUMERO PACIENTES ADULTOS INTERNADOS EM LEITOS UTI  (SUS  PRIVADO)']/rs2['NUMERO LEITOS UTI ADULTO (SUS  PRIVADO)']*100
rs2['uti_a_util'].fillna(0,inplace=True)
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = rs['uti_a_util'].values[0],
    mode = "gauge+number+delta",
    title = {'text': "ICU Ocupation RS - "+str(last_date)},
    delta = {'reference': rs2['uti_a_util'].values[0]},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "green"},
             'steps' : [
                 {'range': [0, 60], 'color': "white"},
                 {'range': [60, 80], 'color': "yellow"},
                 {'range': [80, 100], 'color': "red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': rs['uti_a_util'].values[0]}}))

fig.show()
rs_day = hospital_beds.groupby('just_date').sum()
rs_day['uti_a_util'] = rs_day['NUMERO PACIENTES ADULTOS INTERNADOS EM LEITOS UTI  (SUS  PRIVADO)']/rs_day['NUMERO LEITOS UTI ADULTO (SUS  PRIVADO)']*100
rs_day['uti_a_util'].fillna(0,inplace=True)
rs_day['leitos_d']=100-rs_day['uti_a_util']
layout = Layout(
    title="ICU capacity - RS",
)

fig = go.Figure(data=[
    go.Bar(name='Used', x=rs_day.index, y=round(rs_day['uti_a_util'],2),marker_color='red'),
    go.Bar(name='Available', x=rs_day.index, y=round(rs_day['leitos_d'],2),marker_color='green')

])
fig.update_xaxes(title_text='date')
fig.update_yaxes(title_text='%')
fig.update_layout(barmode='stack')
fig['layout'].update(layout)

fig.show()