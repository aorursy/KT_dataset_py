import pandas as pd
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

from plotly.offline import iplot, init_notebook_mode
import math
import bokeh 
import matplotlib.pyplot as plt
import plotly.express as px
#from urllib.request import urlopen
import json
from dateutil import parser
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.models import Span
import warnings
warnings.filterwarnings("ignore")
output_notebook(resources=INLINE)
country_codes = pd.read_csv('/kaggle/input/countrycodes/countrycodes.csv')
country_codes = country_codes.drop('GDP (BILLIONS)', 1)
country_codes.rename(columns={'COUNTRY': 'Country', 'CODE': 'Code'}, inplace=True)
virus_data = pd.read_csv('')

prev_index = 0
first_time = False
tmp = 0


for i, row in virus_data.iterrows():

    if(virus_data.loc[i,'SNo'] < 1342 and virus_data.loc[i,'Province/State']=='Hubei'):
        if(first_time):
            tmp = virus_data.loc[i,'Confirmed']
            prev_index = i
            virus_data.loc[i,'Confirmed'] = virus_data.loc[i,'Confirmed'] + 593
            first_time = False
        else:
            increment = virus_data.loc[i,'Confirmed'] - tmp
            tmp = virus_data.loc[i,'Confirmed']
            virus_data.loc[i,'Confirmed'] = virus_data.loc[prev_index,'Confirmed'] + increment + 593
            prev_index = i
    

virus_data.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)
virus_data = virus_data.fillna('unknow')
virus_data['Country'] = virus_data['Country'].str.replace('US','United States')
virus_data['Country'] = virus_data['Country'].str.replace('UK','United Kingdom') 
virus_data['Country'] = virus_data['Country'].str.replace('Mainland China','China')
virus_data['Country'] = virus_data['Country'].str.replace('South Korea','Korea, South')
virus_data['Country'] = virus_data['Country'].str.replace('North Korea','Korea, North')
virus_data['Country'] = virus_data['Country'].str.replace('Macau','China')
virus_data['Country'] = virus_data['Country'].str.replace('Ivory Coast','Cote d\'Ivoire')
virus_data = pd.merge(virus_data,country_codes,on=['Country'])
virus_data.head()
#print(len(virus_data))
import plotly.graph_objects as go

total_confirmed = virus_data.loc[virus_data['Date'] == virus_data['Date'].iloc[-1]]
a = total_confirmed.groupby(['Code','Country'])['Confirmed'].sum().reset_index()
fig = go.Figure(data=go.Choropleth(
    locations = a['Code'],
    z = a['Confirmed'],
    text = a['Country'],
    colorscale = 'Viridis',
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'N° cases',
))
fig.update_layout(
    title_text='Total confirmed Coronavirus cases')

fig.show()
formated_gdf = virus_data.groupby(['Date', 'Country'])['Confirmed'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], animation_frame="Date", 
                     title='Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
deaths_inf = virus_data.groupby(['Date'])['Confirmed','Deaths','Recovered'].sum().reset_index()

deaths_day = []
deaths_day.append(deaths_inf['Deaths'][0])
for i in range(1,len(deaths_inf)):
    deaths_day.append(deaths_inf['Deaths'][i] - deaths_inf['Deaths'][i-1])
    
deaths_growth = []
for i in range(len(deaths_day)):
    deaths_growth.append(deaths_day[i] / deaths_inf['Deaths'][i])

datetime = []
a = deaths_inf['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Change in total Deaths of Novel Coronavirus (2019-nCoV)")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Change in total (%)'

p1.line(datetime, deaths_growth, color='#2874A6', 
        legend_label='Growth factor', line_width=1.5)
p1.circle(datetime, deaths_growth, fill_color="black", size=5)
#p1.line(datetime, active_cases['Co-Recov'], color='#FF4500', 
        #legend_label='Sick people without counting recovered', line_width=1.5)
#p1.circle(datetime, active_cases['Co-Recov'], fill_color="black", size=5)

p1.legend.location = 'top_right'

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
dr_countries = virus_data.groupby(['Date'])['Confirmed','Deaths','Recovered'].sum().reset_index()
period = 7
death_rate = []
for i in range(1,len(dr_countries)):
    recover = list(dr_countries['Recovered'])[i] - list(dr_countries['Recovered'])[i-1]
    death = list(dr_countries['Deaths'])[i] - list(dr_countries['Deaths'])[i-1]
    if(recover+death==0):
        death_rate.append(death / (recover+death+1))
    else:
        death_rate.append(death / (recover+death))


p1 = figure(plot_width=600, plot_height=500, title="Death rate of Covid-19 with the novel formula")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days'
p1.yaxis.axis_label = 'Death Rate (%)'

p1.line(np.arange(1,len(death_rate)+1,7), death_rate[::7], color='#2874A6', 
        legend_label='Growth factor', line_width=1.5)
p1.circle(np.arange(1,len(death_rate)+1,7), death_rate[::7], fill_color="black", size=5)

p1.legend.location = 'top_right'

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
new_cases = []

for i in range(1,len(deaths_inf)):
    
    a = list(deaths_inf['Confirmed'])[i-1]
    b = list(deaths_inf['Confirmed'])[i]
             
    new_cases.append(b - a)
    
growth_factor = []

for i in range(1,len(new_cases)):
    
    growth_factor.append(new_cases[i] / new_cases[i-1])
             

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Infection growth factor of COVID-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Factor'

p1.line(datetime, growth_factor, color='#8B4513', 
        legend_label='Growth Factor', line_width=1.5)
p1.circle(datetime, growth_factor, fill_color="black", size=5)
hline = Span(location=1, dimension='width', line_color='red', line_width=1)


p1.legend.location = 'top_right'

output_file("coronavirus.html", title="coronavirus.py")
  
p1.renderers.extend([hline])
show(p1)
time_confirmed = virus_data.groupby('Date')['Confirmed'].sum().reset_index()
time_deaths = virus_data.groupby('Date')['Deaths'].sum().reset_index()
time_recovered = virus_data.groupby('Date')['Recovered'].sum().reset_index()

datetime = []
a = time_confirmed['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Coronavirus infection")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, time_confirmed['Confirmed'], color='#D1BB33', 
        legend_label='Confirmed cases', line_width=1.5)
p1.circle(datetime, time_confirmed['Confirmed'], fill_color="white", size=5)
p1.line(datetime, time_deaths['Deaths'], color='#D1472A', legend_label='Deaths',
       line_width=1.5)
p1.circle(datetime, time_deaths['Deaths'], fill_color="white", size=5)
p1.line(datetime, time_recovered['Recovered'], color='#33A02C', legend_label='Recovered',
       line_width=1.5)
p1.circle(datetime, time_recovered['Recovered'], fill_color="white", size=5)
p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
confirmed_state = virus_data.groupby(['Date','Code'])['Confirmed'].sum().reset_index()
deaths_state = virus_data.groupby(['Date','Code'])['Deaths'].sum().reset_index()
recovered_state = virus_data.groupby(['Date','Code'])['Recovered'].sum().reset_index()

confirmed_china = confirmed_state.loc[confirmed_state['Code'] == 'CHN']
deaths_china = deaths_state.loc[deaths_state['Code'] == 'CHN']
recovered_china = recovered_state.loc[recovered_state['Code'] == 'CHN']

datetime = []
a = confirmed_china['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="China Coronavirus infection")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, confirmed_china['Confirmed'], color='#D1BB33', 
        legend_label='Confirmed cases', line_width=1.5)
p1.circle(datetime, confirmed_china['Confirmed'], fill_color="white", size=5)
p1.line(datetime, deaths_china['Deaths'], color='#D1472A', legend_label='Deaths',
       line_width=1.5)
p1.circle(datetime, deaths_china['Deaths'], fill_color="white", size=5)
p1.line(datetime, recovered_china['Recovered'], color='#33A02C', legend_label='Recovered',
       line_width=1.5)
p1.circle(datetime, recovered_china['Recovered'], fill_color="white", size=5)
p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
confirmed_no_china = confirmed_state.loc[confirmed_state['Code'] != 'CHN']
confirmed_no_china = confirmed_no_china.groupby('Date')['Confirmed'].sum().reset_index()

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Coronavirus infection cases")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, confirmed_china['Confirmed'], color='#D1BB33', 
        legend_label='China', line_width=1.5)
p1.circle(datetime, confirmed_china['Confirmed'], fill_color="white", size=5)
p1.line(datetime, confirmed_no_china['Confirmed'], color='#D1472A', 
        legend_label='Rest of the World',line_width=1.5)
p1.circle(datetime, confirmed_no_china['Confirmed'], fill_color="white", size=5)
p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
regioni_ita = pd.read_csv('/kaggle/input/provinceita/dpc-covid19-ita-province.csv')
#regioni_ita = regioni_ita.loc[regioni_ita['denominazione_regione']!='Lombardia']
last_regioni_ita = regioni_ita.loc[regioni_ita['data'] == regioni_ita['data'].iloc[-1]]
tot_regioni_ita = last_regioni_ita.groupby(['denominazione_regione'])['totale_casi'].sum().reset_index()
tot_regioni_ita = tot_regioni_ita.sort_values('totale_casi', ascending=False)


from bokeh.io import show, output_file
from bokeh.plotting import figure

output_file("bar_stacked.html")

regioni = tot_regioni_ita['denominazione_regione']
infection = ["N° Confirmed"]
colors = ["#CD6155"]

data = {'countries' : regioni,
        'N° Confirmed'   : tot_regioni_ita['totale_casi']}

p = figure(x_range=regioni, plot_height=500, plot_width=700,
           title="COVID-19 infection for Regioni Italiane",
           toolbar_location=None, tools="hover", tooltips="$name @countries: @$name")

p.vbar_stack(infection, x='countries', width=0.9, color=colors, source=data,
             legend_label=infection)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = math.pi/2

output_file("coronavirus.html", title="coronavirus.py")

show(p)
day_regioni_ita = regioni_ita.groupby(['data','denominazione_regione'])['totale_casi'].sum().reset_index()
day_veneto = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Veneto']
day_friuli = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Friuli Venezia Giulia']
day_piemonte = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Piemonte']
day_lombardia = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Lombardia'] 
day_emilia = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Emilia-Romagna']
day_liguria = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Liguria']
day_aosta = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Valle d\'Aosta']

day_toscana = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Toscana']
day_abruzzo = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Abruzzo']
day_marche = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Marche']
day_lazio = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Lazio']
day_umbria = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Umbria']

day_basilicata = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Basilicata']
day_calabria = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Calabria']
day_molise = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Molise']
day_puglia = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Puglia']
day_campania = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Campania']
day_sicilia = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Sicilia']
day_sardegna = day_regioni_ita.loc[day_regioni_ita['denominazione_regione']=='Sardegna']

north_legend = ['Veneto','Friuli Venezia Giulia','Piemonte','Lombardia','Emilia Romagna',
               'Liguria','Valle d\'Aosta']

center_legend = ['Toscana','Abruzzo','Marche','Lazio','Umbria']

south_legend = ['Basilicata','Calabria','Molise','Puglia','Campania','Sicilia','Sardegna']

plt.figure(figsize=(15,10))

data = list(day_veneto['data'])
days = []
for elm in data:
    days.append(elm[:10])
    
sns.set()

plt.plot(days,day_veneto['totale_casi'], marker='o',ms=3)
plt.plot(days,day_friuli['totale_casi'], marker='o',ms=3)
plt.plot(days,day_piemonte['totale_casi'], marker='o',ms=3)
plt.plot(days,day_lombardia['totale_casi'], marker='o',ms=3)
plt.plot(days,day_emilia['totale_casi'], marker='o',ms=3)
plt.plot(days,day_liguria['totale_casi'], marker='o',ms=3)
plt.plot(days,day_aosta['totale_casi'], marker='o',ms=3)

plt.plot(days,day_toscana['totale_casi'], marker='o',ms=3)
plt.plot(days,day_abruzzo['totale_casi'], marker='o',ms=3)
plt.plot(days,day_marche['totale_casi'], marker='o',ms=3)
plt.plot(days,day_lazio['totale_casi'], marker='o',ms=3)
plt.plot(days,day_umbria['totale_casi'], marker='o',ms=3)

plt.plot(days,day_basilicata['totale_casi'], marker='o',ms=3)
plt.plot(days,day_calabria['totale_casi'], marker='o',ms=3)
plt.plot(days,day_molise['totale_casi'], marker='o',ms=3)
plt.plot(days,day_puglia['totale_casi'], marker='o',ms=3)
plt.plot(days,day_campania['totale_casi'], marker='o',ms=3)
plt.plot(days,day_sicilia['totale_casi'], marker='o',ms=3)
plt.plot(days,day_sardegna['totale_casi'], marker='o',ms=3)

plt.ylabel('Number of cases')
plt.xlabel('Date')
plt.xticks(rotation=70)
plt.legend(north_legend + center_legend + south_legend)
#plt.grid()
plt.show()
nazionale_ita = pd.read_csv('/kaggle/input/nazionale/dpc-covid19-ita-andamento-nazionale.csv', error_bad_lines=False)
nazionale_ita.head()
daybyday_cases = []
for i in range(1,len(nazionale_ita['totale_casi'])):
    daybyday_cases.append(list(nazionale_ita['totale_casi'])[i] - list(nazionale_ita['totale_casi'])[i-1])
confirmed_ita = nazionale_ita.groupby(['data'])['totale_casi'].sum().reset_index()
deaths_ita = nazionale_ita.groupby(['data'])['deceduti'].sum().reset_index()
recovered_ita = nazionale_ita.groupby(['data'])['dimessi_guariti'].sum().reset_index()



datetime = []
a = nazionale_ita['data'].to_frame()
for elm in a['data']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Coronavirus infection in Italy")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, confirmed_ita['totale_casi'], color='#C0392B', 
        legend_label='Confirmed cases', line_width=1.5)
p1.circle(datetime, confirmed_ita['totale_casi'], fill_color="white", size=5)
p1.line(datetime, deaths_ita['deceduti'], color='#5DADE2', legend_label='Deaths',
       line_width=1.5)
p1.circle(datetime, deaths_ita['deceduti'], fill_color="white", size=5)
p1.line(datetime, recovered_ita['dimessi_guariti'], color='#E67E22', legend_label='Recovered',
       line_width=1.5)
p1.circle(datetime, recovered_ita['dimessi_guariti'], fill_color="white", size=5)
p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


mean_hm_list = []
mean_hp_list = []
mean_list = []

start = 7
growth_factor_s = growth_factor[start:]

for i in range(len(growth_factor_s)):
    mean, mean_hm, mean_hp = mean_confidence_interval(growth_factor_s[:i], 0.95)
    mean_hm_list.append(mean_hm)
    mean_hp_list.append(mean_hp)
    mean_list.append(mean)
    
mean_hm_list = np.asarray(mean_hm_list)
mean_hp_list = np.asarray(mean_hp_list)
mean_list = np.asarray(mean_list)

mean_range = [i for i in range(len(mean_list))]

plt.figure(figsize=(10,8))
plt.plot(mean_list)
plt.fill_between(mean_range, mean_hm_list, mean_hp_list, 
                 facecolor='b', alpha=0.4, edgecolor='#8F94CC', 
                 linewidth=2, linestyle='dashed')

plt.title("95% - Confidence intervals for the mean R0 in Italy")
plt.legend(["Average R0","CI (95%)"])
plt.ylim(0,3)
plt.show()

hospital_ita = nazionale_ita.groupby(['data'])['ricoverati_con_sintomi'].sum().reset_index()
intensive_ita = nazionale_ita.groupby(['data'])['terapia_intensiva'].sum().reset_index()
home_ita = nazionale_ita.groupby(['data'])['isolamento_domiciliare'].sum().reset_index()



datetime = []
a = nazionale_ita['data'].to_frame()
for elm in a['data']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Conditions of the infected")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, hospital_ita['ricoverati_con_sintomi'], color='#C0392B', 
        legend_label='Mild conditions', line_width=1.5)
p1.circle(datetime, hospital_ita['ricoverati_con_sintomi'], fill_color="white", size=5)
p1.line(datetime, intensive_ita['terapia_intensiva'], color='#5DADE2', legend_label='Serious or Critical',
       line_width=1.5)
p1.circle(datetime, intensive_ita['terapia_intensiva'], fill_color="white", size=5)
p1.line(datetime, home_ita['isolamento_domiciliare'], color='#E67E22', legend_label='Home isolation',
       line_width=1.5)
p1.circle(datetime, home_ita['isolamento_domiciliare'], fill_color="white", size=5)
p1.legend.location = "top_left"

hline = Span(location=6100, dimension='width', name='Intensive care places in Italy', 
             line_color='blue', line_width=1)

  
p1.renderers.extend([hline])

output_file("coronavirus.html", title="coronavirus.py")


show(p1)
deaths = nazionale_ita['deceduti']

death_rate = []
for i in range(1,len(deaths)):
    death_rate.append((deaths[i] - deaths[i-1]) / deaths[i-1])

data = list(day_veneto['data'])
datetime = []
for elm in data:
    datetime.append(elm[:10])
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Coronavirus n° deaths in Italy")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of deaths'

p1 = figure(plot_width=600, plot_height=500, title="Death growth % day by day")
p1.line(np.arange(0,len(death_rate),1), death_rate, color='#C0392B', 
        legend_label='Death Rate', line_width=1.5)
p1.circle(np.arange(0,len(death_rate),1), death_rate, fill_color="white", size=5)
p1.xaxis.axis_label = 'Number of days'
p1.yaxis.axis_label = 'Rate'

p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
day_swabs = []

for i in range(1,len(nazionale_ita)):
    day_swabs.append(list(nazionale_ita['tamponi'])[i]- list(nazionale_ita['tamponi'])[i-1])

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
# Shade the area between y1 and line y=0
plt.figure(figsize=(10,8))
plt.fill_between(np.arange(0,len(day_swabs)), day_swabs, 0,
                 facecolor="orange", # The fill color
                 color='blue',       # The outline color
                 alpha=0.5)          # Transparency of the fill
plt.fill_between(np.arange(0,len(daybyday_cases)), daybyday_cases, 0,
                 facecolor="orange", # The fill color
                 color='orange',       # The outline color
                 alpha=0.5)          # Transparency of the fill
plt.legend(['Swabs','New cases'],loc=2)
plt.xticks(np.arange(0,len(daybyday_cases),2))
plt.xlabel('Days')
plt.ylabel('Number')
#plt.plot(running_mean(day_swabs,5))
# Show the plot
plt.show()
active_cases = virus_data.groupby(['Date'])['Confirmed','Deaths','Recovered'].sum().reset_index()
active_cases['Co-Deaths'] = active_cases['Confirmed'] - active_cases['Deaths']
active_cases['Co-Recov'] = active_cases['Confirmed'] - active_cases['Recovered']
active_cases['Active'] = active_cases['Confirmed'] - active_cases['Deaths'] - active_cases['Recovered']

datetime = []
a = active_cases['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Active cases of COVID-19 (Confirmed - Recovered - Deaths)")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Active cases'

p1.line(datetime, active_cases['Active'], color='#8B4513', 
        legend_label='Sick people without counting deaths & recovered', line_width=1.5)
p1.circle(datetime, active_cases['Active'], fill_color="black", size=5)
#p1.line(datetime, active_cases['Co-Deaths'], color='#FFA500', 
        #legend_label='Sick people without counting deaths', line_width=1.5)
#p1.circle(datetime, active_cases['Co-Deaths'], fill_color="black", size=5)
#p1.line(datetime, active_cases['Co-Recov'], color='#FF4500', 
        #legend_label='Sick people without counting recovered', line_width=1.5)
#p1.circle(datetime, active_cases['Co-Recov'], fill_color="black", size=5)

p1.legend.location = 'bottom_right'

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
mortality_rate = virus_data.groupby(['Date'])['Confirmed','Deaths'].sum().reset_index()
mortality_rate['Rate'] = mortality_rate['Deaths'] / mortality_rate['Confirmed']
mortality_rate['Infection'] = mortality_rate['Confirmed'] / mortality_rate['Confirmed'].max()

datetime = []
a = mortality_rate['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Mortality rate of COVID-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Mortality (%)'

p1.line(datetime, mortality_rate['Rate'], color='#900C3F', 
        legend_label='Mortality Rate', line_width=1.5)
p1.circle(datetime, mortality_rate['Rate'], fill_color="black", size=5)

p2 = figure(plot_width=600, plot_height=200, title="Normalized histogram of infections")
p2.vbar(x=np.arange(0,len(datetime),1), top=mortality_rate['Infection'], 
        width=0.7, bottom=0, color="firebrick")
p2.xaxis.visible = False
p1.xaxis.axis_label = 'Infections over time'
p1.yaxis.axis_label = 'Deaths (%)'

p1.legend.location = 'bottom_right'

output_file("coronavirus.html", title="coronavirus.py")

show(column(p1,p2))
healed_rate = virus_data.groupby(['Date'])['Confirmed','Recovered'].sum().reset_index()
healed_rate['Rate'] = healed_rate['Recovered'] / healed_rate['Confirmed']
healed_rate['Infection'] = healed_rate['Confirmed'] / healed_rate['Confirmed'].max()

datetime = []
a = mortality_rate['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", 
            title="Recovered people rate of COVID-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Recovered (%)'

p1.line(datetime, healed_rate['Rate'], color='#498748', 
        legend_label='Recovered people Rate', line_width=1.5)
p1.circle(datetime, healed_rate['Rate'], fill_color="black", size=5)

p2 = figure(plot_width=600, plot_height=200, title="Normalized histogram of infections")
p2.vbar(x=np.arange(0,len(datetime),1), top=healed_rate['Infection'], 
        width=0.7, bottom=0, color="firebrick")
p2.xaxis.visible = False
p1.xaxis.axis_label = 'Infections over time'
p1.yaxis.axis_label = 'Recovered (%)'

p1.legend.location = 'top_left'

output_file("coronavirus.html", title="coronavirus.py")

show(column(p1,p2))
datetime = []
a = mortality_rate['Date'].to_frame()
for elm in a['Date']:   
    b = elm[0:10]
    datetime.append(b)
    
datetime = pd.to_datetime(datetime)

p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", 
            title="Recovered / Deaths people rate of COVID-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Percentual'

p1.line(datetime, mortality_rate['Rate'], color='#900C3F', 
        legend_label='Mortality Rate', line_width=1.5)
p1.circle(datetime, mortality_rate['Rate'], fill_color="black", size=5)

p1.line(datetime, healed_rate['Rate'], color='#498748', 
        legend_label='Recovered people Rate', line_width=1.5)
p1.circle(datetime, healed_rate['Rate'], fill_color="black", size=5)

p1.legend.location = 'top_left'

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
country_cases = virus_data.groupby('Code')['Confirmed','Deaths','Recovered'].max().reset_index()
country_cases = country_cases.sort_values('Confirmed', ascending=False)
country_cases = country_cases[:50]
#country_cases['Confirmed'] = np.log10(country_cases['Confirmed']+1)
#country_cases['Deaths'] = np.log10(country_cases['Deaths']+1)
#country_cases['Recovered'] = np.log10(country_cases['Recovered']+1)

from bokeh.io import show, output_file
from bokeh.plotting import figure

output_file("bar_stacked.html")

countries = country_cases['Code']
infection = ["Confirmed", "Deaths", "Recovered"]
colors = ["#CD6155", "#F8C471", "#34495E"]

data = {'countries' : countries,
        'Confirmed'   : country_cases['Confirmed'],
        'Deaths'   : country_cases['Deaths'],
        'Recovered'   : country_cases['Recovered']}

p = figure(x_range=countries, plot_height=650, plot_width=800,
           title="Logarithmic scale of COVID-19 infection for top-50 countries for infections",
           toolbar_location=None, tools="hover", tooltips="$name @countries: @$name")

p.vbar_stack(infection, x='countries', width=0.9, color=colors, source=data,
             legend_label=infection)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = math.pi/2

show(p)
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

fb_virus_data = virus_data
fb_virus_data = fb_virus_data.groupby('Date')['Confirmed'].sum().reset_index()
# Prophet requires columns ds (Date) and y (value)
fb_confirmed = fb_virus_data[["Date","Confirmed"]]
fb_confirmed = fb_confirmed.rename(columns={'Date': 'ds', 'Confirmed': 'y'})
# Make the prophet model and fit on the data
changepoint_prior_scale = [0.05,0.1,0.15,0.2,0.25]

model = Prophet(seasonality_mode = 'additive', changepoint_prior_scale=0.25)
model.fit(fb_confirmed)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
#figure = model.plot(forecast)
#axes = figure.get_axes()
#axes[0].set_xlabel('Date')
#axes[0].set_ylabel('Confirmed cases forecast')

dates = []
for elm in fb_confirmed.ds:
    a = elm[6:]
    b = elm[:2]
    c = elm[3:5]
    d = a+'-'+b+'-'+c
    dates.append(d)


trace1 = {
  "fill": None, 
  "mode": "markers",
  "marker_size": 10,
  "name": "n° of Confirmed", 
  "type": "scatter", 
  "x": dates, 
  "y": fb_confirmed.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": forecast.ds, 
  "y": forecast.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": forecast.ds, 
  "y": forecast.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers",
  "marker_size": 4,
  "name": "prediction", 
  "type": "scatter", 
  "x": forecast.ds, 
  "y": forecast.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed cases - Time Series Forecast", 
  "xaxis": {
    "title": "Daily Dates", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed cases", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)
    
yhat = list(forecast['yhat'][:-7])
y = list(fb_virus_data['Confirmed'])

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('Mean absolute percentage error: ', mape(y,yhat))
recovered_rate = []

for i in range(1,len(recovered_ita)):
    
    x = list(recovered_ita['dimessi_guariti'])[i]
    y = list(recovered_ita['dimessi_guariti'])[i-1]
    if y==0 and x==0:
        recovered_rate.append(0)
    elif(y==0):
        recovered_rate.append(x/x)
    else:
        z = (x - y) / x
        recovered_rate.append(z)
 
print('Recovered rate in Italy: ', np.mean(recovered_rate))
# Total population, N.
Ro = 2.5

d = 1 / np.mean(recovered_rate)
gamma = 0.1 #np.mean(recovered_rate)
beta = Ro * gamma

N = 60000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 2, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = beta, gamma 
# A grid of time points (in days)
t = np.linspace(0, 180, 180)

# The SIR model differential equations.
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I /N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Integrate the SIR equations over the time grid, t.
solution = odeint(sir_model, [S0, I0, R0], t, args=(N, beta, gamma))
soultion = np.array(solution)

p1 = figure(plot_width=600, plot_height=500, title="SIR Model for Coronavirus (2019-nCoV)")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days from 27/01/20 (Estimate of the first cases in Italy)'
p1.yaxis.axis_label = 'Population'

p1.line(t, solution[:,0], color='#D35400', 
        legend_label='Susceptible', line_width=1.5)
p1.circle(t, solution[:,0], fill_color="black", size=1)

p1.line(t, solution[:,1], color='#2E4053', 
        legend_label='Infected', line_width=1.5)
p1.circle(t, solution[:,1], fill_color="black", size=1)

p1.line(t, solution[:,2], color='#28B463', 
       legend_label='Recovered', line_width=1.5)
p1.circle(t, solution[:,2], fill_color="black", size=1)

p1.legend.location = 'top_right'

show(p1)
# Total population, N.
Ro = 2.8

#d = 1 / np.mean(recovered_rate)
gamma = 1/10
mu = 0.
alpha = 1/5
beta = Ro*((mu + alpha)*(mu+gamma))/alpha 
Blambda = 0.
#print(beta)

N = 60000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0, E0 = 500, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0
# A grid of time points (in days)
t = np.linspace(0, 200, 200)

# The SIR model differential equations.
def seir_model(y, t, N, beta, gamma, alpha, Blambda, mu):
    S, E, I, R = y
    dSdt = Blambda -mu*S -beta*S*I/N
    dEdt = beta*S*I/N - (mu+alpha)*E
    dIdt = alpha*E - (gamma+mu)*I
    dRdt = gamma*I - mu*R
    return dSdt, dEdt, dIdt, dRdt

# Integrate the SIR equations over the time grid, t.
solution2 = odeint(seir_model, [S0, E0, I0, R0], t, args=(N, beta, gamma, alpha, Blambda, mu))
soultion2 = np.array(solution2)

p1 = figure(plot_width=600, plot_height=500, title="SEIR Model for Coronavirus (2019-nCoV)")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days from 27/01/20 (Estimate of the first cases in Italy)'
p1.yaxis.axis_label = 'Population'

p1.line(t, solution2[:,0], color='#D35400', 
        legend_label='Susceptible', line_width=1.5)
p1.circle(t, solution2[:,0], fill_color="black", size=1)

p1.line(t, solution2[:,1], color='#2E4053', 
        legend_label='Exposed', line_width=1.5)
p1.circle(t, solution2[:,1], fill_color="black", size=1)

p1.line(t, solution2[:,2], color='#28B463', 
       legend_label='Infected', line_width=1.5)
p1.circle(t, solution2[:,2], fill_color="black", size=1)

p1.line(t, solution2[:,3], color='#821063', 
       legend_label='Recovered', line_width=1.5)
p1.circle(t, solution2[:,3], fill_color="black", size=1)

p1.legend.location = 'top_right'

show(p1)
# Total population, N.
Ro = 2.8

#d = 1 / np.mean(recovered_rate)
gamma = 1/10
mu = 0.
alpha = 1/5
beta = Ro*((mu + alpha)*(mu+gamma))/alpha 
Blambda = 0.
#print(beta)

N = 60000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0, E0 = 500, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0
# A grid of time points (in days)
t = np.linspace(0, 360, 360)

# The SIR model differential equations.
def seir_model_social(y, t, N, ro, beta, gamma, alpha, Blambda, mu):
    S, E, I, R = y
    dSdt = Blambda -mu*S -ro*beta*S*I/N
    dEdt = ro*beta*S*I/N - (mu+alpha)*E
    dIdt = alpha*E - (gamma+mu)*I
    dRdt = gamma*I - mu*R
    return dSdt, dEdt, dIdt, dRdt

# Integrate the SIR equations over the time grid, t.
ro1=1
solution1 = odeint(seir_model_social, [S0, E0, I0, R0], t, args=(N, ro1, beta, gamma, alpha, Blambda, mu))
soultion1 = np.array(solution1)

ro2=0.8
solution2 = odeint(seir_model_social, [S0, E0, I0, R0], t, args=(N, ro2, beta, gamma, alpha, Blambda, mu))
soultion2 = np.array(solution2)

ro3=0.6
solution3 = odeint(seir_model_social, [S0, E0, I0, R0], t, args=(N, ro3, beta, gamma, alpha, Blambda, mu))
soultion3 = np.array(solution3)

p1 = figure(plot_width=600, plot_height=500, title="SEIR Model for Coronavirus (2019-nCoV) with social distance")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days from 27/01/20 (Estimate of the first cases in Italy)'
p1.yaxis.axis_label = 'Population'

p1.line(t, solution1[:,2], color='#D35400', 
        legend_label='Infected ρ=1', line_width=1.5)
p1.circle(t, solution1[:,2], fill_color="black", size=1)
p1.line(t, solution1[:,1], color='#D35400', 
        legend_label='Exposed ρ=1', line_dash="4 4", line_width=0.5)
p1.circle(t, solution1[:,1], fill_color="black", line_dash="4 4", size=0.02)

p1.line(t, solution2[:,2], color='#2E4053', 
        legend_label='Infected ρ=0.8', line_width=1.5)
p1.circle(t, solution2[:,2], fill_color="black", size=1)
p1.line(t, solution2[:,1], color='#2E4053', 
        legend_label='Exposed ρ=0.8', line_dash="4 4", line_width=0.5)
p1.circle(t, solution2[:,1], fill_color="black", line_dash="4 4", size=0.02)

p1.line(t, solution3[:,2], color='#28B463', 
       legend_label='Infected ρ=0.6', line_width=1.5)
p1.circle(t, solution3[:,2], fill_color="black", size=1)
p1.line(t, solution3[:,1], color='#28B463', 
       legend_label='Exposed ρ=0.6', line_dash="4 4", line_width=0.5)
p1.circle(t, solution3[:,1], fill_color="black", line_dash="4 4", size=0.02)

p1.legend.location = 'top_right'

show(p1)
# Total population, N.
Ro = 2.78

#d = 1 / np.mean(recovered_rate)
gamma = 1/10
alpha = 0.067
eta = alpha
beta = 0.373
#print(beta)

N = 60000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, Q0, R0 = 500, 250, 250
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0
# A grid of time points (in days)
t = np.linspace(0, 200, 200)

# The SIR model differential equations.
def siqr_model(y, t, N, beta, gamma, alpha, eta):
    S, I, Q, R = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - (alpha+eta)*I
    dQdt = eta*I - gamma*Q
    dRdt = gamma*Q
    return dSdt, dIdt, dQdt, dRdt


# Integrate the SIR equations over the time grid, t.
solution2 = odeint(siqr_model, [S0, I0, Q0, R0], t, args=(N, beta, gamma, alpha, eta))
soultion2 = np.array(solution2)

p1 = figure(plot_width=600, plot_height=500, title="SIQR Model for Coronavirus (2019-nCoV)")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days from 27/01/20 (Estimate of the first cases in Italy)'
p1.yaxis.axis_label = 'Population'

p1.line(t, solution2[:,0], color='#D35400', 
        legend_label='Susceptible', line_width=1.5)
p1.circle(t, solution2[:,0], fill_color="black", size=1)

p1.line(t, solution2[:,1], color='#2E4053', 
        legend_label='Infected', line_width=1.5)
p1.circle(t, solution2[:,1], fill_color="black", size=1)

p1.line(t, solution2[:,2], color='#28B463', 
       legend_label='Quarantined', line_width=1.5)
p1.circle(t, solution2[:,2], fill_color="black", size=1)

p1.line(t, solution2[:,3], color='#821063', 
       legend_label='Recovered', line_width=1.5)
p1.circle(t, solution2[:,3], fill_color="black", size=1)

p1.legend.location = 'top_right'

show(p1)
x = np.linspace(0,100,1000)
L = 18800
k = 0.137
a = 35
y =  L / ( 1 + np.exp(-k*(x-a)) ) 

daybyday_veneto = []
for i in range(1,len(list(day_veneto['totale_casi']))):
    daybyday_veneto.append(list(day_veneto['totale_casi'])[i] -  list(day_veneto['totale_casi'])[i-1])
    

growth_veneto = []

for i in range(1,len(daybyday_veneto)):
    
    if(daybyday_veneto==0):
        continue
    else:
        growth_veneto.append(daybyday_veneto[i] / daybyday_veneto[i-1])

xm = np.argmax(daybyday_veneto) + 5.5
T = 0.0001
L1 = 18000
k = 0.085
y1 = L1 / (1 + T*np.exp(-k*(x-xm)))**(1/T)


p1 = figure(plot_width=600, plot_height=500, title="Logistic curve of Veneto")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Days from first cases in reported by Protezione Civile)'
p1.yaxis.axis_label = 'Population'

p1.line(x, y, color='#D35400', 
        legend_label='Logistic curve', line_width=1.5)
p1.circle(np.arange(0,len(list(day_veneto['totale_casi'])),1),
          list(day_veneto['totale_casi']), fill_color="black", size=3)

p1.line(x, y1, color='#B91422', 
        legend_label='Logistic curve asymmetrical', line_dash="4 4",line_width=0.8)

p1.legend.location = 'bottom_right'
show(p1)
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


mean_hm_list = []
mean_hp_list = []
mean_list = []

start = 7
growth_veneto_s = growth_veneto[start:]

for i in range(len(growth_veneto_s)):
    mean, mean_hm, mean_hp = mean_confidence_interval(growth_veneto_s[:i], 0.95)
    mean_hm_list.append(mean_hm)
    mean_hp_list.append(mean_hp)
    mean_list.append(mean)
    
mean_hm_list = np.asarray(mean_hm_list)
mean_hp_list = np.asarray(mean_hp_list)
mean_list = np.asarray(mean_list)

mean_range = [i for i in range(len(mean_list))]

plt.figure(figsize=(10,8))
plt.plot(mean_list)
plt.fill_between(mean_range, mean_hm_list, mean_hp_list, 
                 facecolor='b', alpha=0.4, edgecolor='#8F94CC', 
                 linewidth=2, linestyle='dashed')

plt.title("95% - Confidence intervals for the mean R0 in Veneto")
plt.legend(["Average R0","CI (95%)"])
plt.ylabel('Average R0')
plt.xlabel('Number of days')
plt.ylim(0,3)
plt.show()
veneto_det = pd.read_csv('/kaggle/input/regioni/dpc-covid19-ita-regioni.csv')
veneto_det = veneto_det.loc[veneto_det['denominazione_regione']=='Veneto']

variazione_morti = []

for i in range(1,len(veneto_det)):
    variazione_morti.append(list(veneto_det['deceduti'])[i] - list(veneto_det['deceduti'])[i-1])
    
tamponi_veneto = []

for i in range(1,len(veneto_det)):
    tamponi_veneto.append(list(veneto_det['tamponi'])[i] - list(veneto_det['tamponi'])[i-1])
    
guariti_veneto = []

for i in range(1,len(veneto_det)):
    guariti_veneto.append(list(veneto_det['dimessi_guariti'])[i] - list(veneto_det['dimessi_guariti'])[i-1])
print("DATI ELABORATI PER L'ITALIA (RIFERIMENTO PROTEZIONE CIVILE)")
print("\n")
print('Numero totale di contagi in Italia ad oggi: ', list(confirmed_ita['totale_casi'])[-1])
print('Variazione percentuale nuovi positivi: ', 
      np.round(((list(confirmed_ita['totale_casi'])[-1] - list(confirmed_ita['totale_casi'])[-2]) / list(confirmed_ita['totale_casi'])[-1])*100,2),'%')
print('Contagi in tutta Italia oggi: ', list(nazionale_ita['totale_casi'])[-1] - list(nazionale_ita['totale_casi'])[-2])
print("Variazione nuovi positivi: ", list(nazionale_ita['variazione_totale_positivi'])[-1])
print("Rapporto tra i contagi di oggi e quelli di ieri (R0): ", np.round(growth_factor[-1],3))
print("Totale dei morti in tutta Italia: ", list(deaths_ita['deceduti'])[-1])
print("Morti in tutta Italia oggi: ", list(deaths_ita['deceduti'])[-1] - list(deaths_ita['deceduti'])[-2])
print("Crescita dei morti rispetto a ieri in tutta Italia: ", np.round(death_rate[-1]*100,2),'%')
print("Totale dei guariti in tutta Italia ad oggi: ", list(recovered_ita['dimessi_guariti'])[-1])
print("Guariti in tutta Italia oggi: ", list(recovered_ita['dimessi_guariti'])[-1] - list(recovered_ita['dimessi_guariti'])[-2])
print("Numero di persone ricoverate con sintomi attualmente: ",
      list(hospital_ita['ricoverati_con_sintomi'])[-1])
print("Numero di persone ricoverate in terapia intensiva attualmente: ",
      list(intensive_ita['terapia_intensiva'])[-1])
print("Numero di persone in isolamento domiciliare attualmente: ",
      list(home_ita['isolamento_domiciliare'])[-1])
print("Numero dei tamponi effettutuati oggi: ",
      list(day_swabs)[-1])

print("\n") 
print("DATI ELABORATI PER IL VENETO (RIFERIMENTO PROTEZIONE CIVILE)")
print("\n")
print('Numero di contagi totali in Veneto: ', list(day_veneto['totale_casi'])[-1])
print('Contagi in Veneto oggi: ', list(day_veneto['totale_casi'])[-1] - list(day_veneto['totale_casi'])[-2])
#print('Variazione nuovi positivi: ', list(veneto_det['variazione_totale_positivi'])[-1])
print('Variazione percentuale nuovi positivi: ', 
      np.round(((list(day_veneto['totale_casi'])[-1] - list(day_veneto['totale_casi'])[-2]) / list(day_veneto['totale_casi'])[-1])*100,2),'%')
print("Morti in totale: ", np.sum(variazione_morti))
print("Morti di oggi: ", variazione_morti[-1])
print("Numero dei tamponi effettutuati oggi:", tamponi_veneto[-1])
print("Guariti in totale: ", np.sum(guariti_veneto))
print("Guariti di oggi: ", guariti_veneto[-1])
print("Rapporto tra i contagi di oggi e quelli di ieri (R0): ", np.round(growth_veneto[-1],3))
p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Condizioni degli infetti in Veneto")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, veneto_det['ricoverati_con_sintomi'], color='#C0392B', 
        legend_label='Ricoverati con sintomi', line_width=1.5)
p1.circle(datetime, veneto_det['ricoverati_con_sintomi'], fill_color="white", size=5)
p1.line(datetime, veneto_det['terapia_intensiva'], color='#5DADE2', legend_label='Terapia intensiva',
       line_width=1.5)
p1.circle(datetime, veneto_det['terapia_intensiva'], fill_color="white", size=5)
p1.line(datetime, veneto_det['isolamento_domiciliare'], color='#E67E22', legend_label='Isolamento',
       line_width=1.5)
p1.circle(datetime, veneto_det['isolamento_domiciliare'], fill_color="white", size=5)
p1.legend.location = "top_left"


output_file("coronavirus.html", title="coronavirus.py")


show(p1)
p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Condizioni degli infetti in Veneto")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, veneto_det['totale_positivi'], color='#4C31C8', 
        legend_label='Attualmente positivi', line_width=1.5)
p1.circle(datetime, veneto_det['totale_positivi'], fill_color="white", size=4)

p1.line(datetime, veneto_det['deceduti'], color='#C0392B', 
        legend_label='Deceduti', line_width=1.5)
p1.circle(datetime, veneto_det['deceduti'], fill_color="white", size=4)

p1.line(datetime, veneto_det['dimessi_guariti'], color='#2635A5', 
        legend_label='Dimessi/Guariti', line_width=1.5)
p1.circle(datetime, veneto_det['dimessi_guariti'], fill_color="white", size=4)




output_file("coronavirus.html", title="coronavirus.py")

p1.legend.location = "top_left"
show(p1)
p1 = figure(plot_width=600, plot_height=500, x_axis_type="datetime", title="Variazione positivi in Italia")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Number of cases'

p1.line(datetime, nazionale_ita['variazione_totale_positivi'], color='#C0392B', 
        legend_label='Variazione nuovi positivi', line_width=1.5)
p1.circle(datetime, nazionale_ita['variazione_totale_positivi'], fill_color="white", size=5)

output_file("coronavirus.html", title="coronavirus.py")

p1.legend.location = "top_left"
show(p1)
period = 1

regioni = pd.read_csv('/kaggle/input/regioni/dpc-covid19-ita-regioni.csv')

veneto = regioni.loc[regioni['denominazione_regione']=='Veneto']
veneto_positivi = list(veneto['nuovi_positivi'])[::period]
veneto_casi = list(veneto['totale_casi'])[::period]

veneto_positivi = [0 if x<0 else x for x in veneto_positivi]


emilia = regioni.loc[regioni['denominazione_regione']=='Emilia-Romagna']
emilia_positivi = list(emilia['nuovi_positivi'])[::period]
emilia_casi = list(emilia['totale_casi'])[::period]

emilia_positivi = [0 if x<0 else x for x in emilia_positivi]
    
    
lomba = regioni.loc[regioni['denominazione_regione']=='Lombardia']
lomba_positivi = list(lomba['nuovi_positivi'])[::period]
lomba_casi = list(lomba['totale_casi'])[::period]

lomba_positivi = [0 if x<0 else x for x in lomba_positivi]
  

piemonte = regioni.loc[regioni['denominazione_regione']=='Piemonte']
piemonte_positivi = list(piemonte['nuovi_positivi'])[::period]
piemonte_casi = list(piemonte['totale_casi'])[::period]

piemonte_positivi = [0 if x<0 else x for x in piemonte_positivi]


toscana = regioni.loc[regioni['denominazione_regione']=='Toscana']
toscana_positivi = list(toscana['nuovi_positivi'])[::period]
toscana_casi = list(toscana['totale_casi'])[::period]

toscana_positivi = [0 if x<0 else x for x in toscana_positivi]


marche = regioni.loc[regioni['denominazione_regione']=='Marche']
marche_positivi = list(marche['nuovi_positivi'])[::period]
marche_casi = list(marche['totale_casi'])[::period] 

marche_positivi = [0 if x<0 else x for x in marche_positivi]
    
    
friuli = regioni.loc[regioni['denominazione_regione']=='Friuli Venezia Giulia']
friuli_positivi = list(friuli['nuovi_positivi'])[::period]
friuli_casi = list(friuli['totale_casi'])[::period]

friuli_positivi = [0 if x<0 else x for x in friuli_positivi]

    
exponential_line_x = []
exponential_line_y = []
for i in range(10):
    exponential_line_x.append(i)
    exponential_line_y.append(i)

p1 = figure(plot_width=800, plot_height=550, title="Trajectory of Covid-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Total number of detected cases (Log scale)'
p1.yaxis.axis_label = 'New confirmed cases (Log scale)'

p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=0.5)
p1.line(np.log(friuli_casi), np.log(friuli_positivi), color='#DBAE23', 
        legend_label='Friuli Venezia Giulia', line_width=1)
p1.circle(np.log(friuli_casi), np.log(friuli_positivi), fill_color="white", size=2)

p1.line(np.log(emilia_casi), np.log(emilia_positivi), color='#3EC358', 
        legend_label='Emilia Romagna', line_width=1)
p1.circle(np.log(emilia_casi), np.log(emilia_positivi), fill_color="white", size=2)

p1.line(np.log(veneto_casi), np.log(veneto_positivi), color='#3E4CC3', 
        legend_label='Veneto', line_width=1)
p1.circle(np.log(veneto_casi), np.log(veneto_positivi), fill_color="white", size=2)

p1.line(np.log(piemonte_casi), np.log(piemonte_positivi), color='#F54138', 
        legend_label='Piemonte', line_width=1)
p1.circle(np.log(piemonte_casi), np.log(piemonte_positivi), fill_color="white", size=2)

p1.line(np.log(marche_casi), np.log(marche_positivi), color='#23BCDB', 
        legend_label='Marche', line_width=1)
p1.circle(np.log(marche_casi), np.log(marche_positivi), fill_color="white", size=2)

p1.line(np.log(toscana_casi), np.log(toscana_positivi), color='#010A0C', 
        legend_label='Toscana', line_width=1)
p1.circle(np.log(toscana_casi), np.log(toscana_positivi), fill_color="white", size=2)

p1.line(np.log(lomba_casi), np.log(lomba_positivi), color='#017A0C', 
        legend_label='Lombardia', line_width=1)
p1.circle(np.log(lomba_casi), np.log(lomba_positivi), fill_color="white", size=2)

p1.legend.location = "top_left"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)
period = 2

regioni = pd.read_csv('/kaggle/input/regioni/dpc-covid19-ita-regioni.csv')

veneto = regioni.loc[regioni['denominazione_regione']=='Veneto']
veneto_positivi = list(veneto['nuovi_positivi'])[::period]
veneto_casi = list(veneto['totale_casi'])[::period]

veneto_positivi = [0 if x<0 else x for x in veneto_positivi]


emilia = regioni.loc[regioni['denominazione_regione']=='Emilia-Romagna']
emilia_positivi = list(emilia['nuovi_positivi'])[::period]
emilia_casi = list(emilia['totale_casi'])[::period]

emilia_positivi = [0 if x<0 else x for x in emilia_positivi]
    
    
lomba = regioni.loc[regioni['denominazione_regione']=='Lombardia']
lomba_positivi = list(lomba['nuovi_positivi'])[::period]
lomba_casi = list(lomba['totale_casi'])[::period]

lomba_positivi = [0 if x<0 else x for x in lomba_positivi]
  

piemonte = regioni.loc[regioni['denominazione_regione']=='Piemonte']
piemonte_positivi = list(piemonte['nuovi_positivi'])[::period]
piemonte_casi = list(piemonte['totale_casi'])[::period]

piemonte_positivi = [0 if x<0 else x for x in piemonte_positivi]


toscana = regioni.loc[regioni['denominazione_regione']=='Toscana']
toscana_positivi = list(toscana['nuovi_positivi'])[::period]
toscana_casi = list(toscana['totale_casi'])[::period]

toscana_positivi = [0 if x<0 else x for x in toscana_positivi]


marche = regioni.loc[regioni['denominazione_regione']=='Marche']
marche_positivi = list(marche['nuovi_positivi'])[::period]
marche_casi = list(marche['totale_casi'])[::period] 

marche_positivi = [0 if x<0 else x for x in marche_positivi]
    
    
friuli = regioni.loc[regioni['denominazione_regione']=='Friuli Venezia Giulia']
friuli_positivi = list(friuli['nuovi_positivi'])[::period]
friuli_casi = list(friuli['totale_casi'])[::period]

friuli_positivi = [0 if x<0 else x for x in friuli_positivi]

    
exponential_line_x = []
exponential_line_y = []
for i in range(10):
    exponential_line_x.append(i)
    exponential_line_y.append(i)

p1 = figure(plot_width=800, plot_height=550, title="Trajectory of Covid-19")
p1.grid.grid_line_alpha=0.3
p1.ygrid.band_fill_color = "olive"
p1.ygrid.band_fill_alpha = 0.1
p1.xaxis.axis_label = 'Total number of detected cases'
p1.yaxis.axis_label = 'New confirmed cases'

#p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=0.5)
p1.line(friuli_casi, friuli_positivi, color='#DBAE23', 
        legend_label='Friuli Venezia Giulia', line_width=1)
p1.circle(friuli_casi, friuli_positivi, fill_color="white", size=2)

p1.line(emilia_casi, emilia_positivi, color='#3EC358', 
        legend_label='Emilia Romagna', line_width=1)
p1.circle(emilia_casi, emilia_positivi, fill_color="white", size=2)

p1.line(veneto_casi, veneto_positivi, color='#3E4CC3', 
        legend_label='Veneto', line_width=1)
p1.circle(veneto_casi, veneto_positivi, fill_color="white", size=2)

p1.line(piemonte_casi, piemonte_positivi, color='#F54138', 
        legend_label='Piemonte', line_width=1)
p1.circle(piemonte_casi, piemonte_positivi, fill_color="white", size=2)

p1.line(marche_casi, marche_positivi, color='#23BCDB', 
        legend_label='Marche', line_width=1)
p1.circle(marche_casi, marche_positivi, fill_color="white", size=2)

p1.line(toscana_casi, toscana_positivi, color='#010A0C', 
        legend_label='Toscana', line_width=1)
p1.circle(toscana_casi, toscana_positivi, fill_color="white", size=2)

p1.line(lomba_casi, lomba_positivi, color='#017A0C', 
        legend_label='Lombardia', line_width=1)
p1.circle(lomba_casi, lomba_positivi, fill_color="white", size=2)

#p1.legend.location = "bottom_right"

output_file("coronavirus.html", title="coronavirus.py")

show(p1)


