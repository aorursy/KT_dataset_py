import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as po
import requests
import json
import ipywidgets as widgets
from IPython.display import display as disp
from ipywidgets import interact, interact_manual
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

%matplotlib inline

#loading csvs to dataframes
def update_data():
    print('Fetching data...')
    try:
        ages = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_Altersgruppe.csv', sep=';', decimal=',')
        gkz = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_GKZ.csv', sep=';', decimal=',')
        tsbl = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_Timeline.csv', sep=';', decimal=',')
        tsbez = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_Timeline_GKZ.csv', sep=';', decimal=',')
        hosptes = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFallzahlen.csv', sep=';', decimal=',')
        mortvie = pd.read_csv('https://www.wien.gv.at/gogv/l9ogdmortalitaetmonatlich', sep=';', decimal=',')

        print('All data loaded successfully')
    except Exception as e:
        print(f'Something went wrong: {e}')
        
    return (ages, gkz, tsbl, tsbez, hosptes, mortvie)

    # json requests are slow - commenting them out for now
    #bez = requests.get('https://corona-ampel.gv.at/sites/corona-ampel.gv.at/files/assets/Warnstufen_Corona_Ampel_aktuell.json')
    #bezjson = bez.json()
    #gem = requests.get('https://corona-ampel.gv.at/sites/corona-ampel.gv.at/files/assets/Warnstufen_Corona_Ampel_Gemeinden_aktuell.json')
    #gemjson = gem.json()

def make_df_bl(tsbl, hosptes):
    # joining Tests & Hospital data to timeseries by Bundesland
    tsbl['joindate'] = pd.to_datetime(tsbl.Time, infer_datetime_format=True).dt.date
    hosptes['joindate']=pd.to_datetime(hosptes.MeldeDatum, dayfirst=True).dt.date
    # gemeinsmer Dataframe aus Hospitalisierten
    df_bl = pd.merge(tsbl, hosptes, left_on=['joindate', 'BundeslandID'], right_on=['joindate','BundeslandID'],suffixes=('','_hosptes') )
    # Active Cases missing from Dataset - added manually
    df_bl['FaelleAktiv'] = df_bl.AnzahlFaelleSum-df_bl.AnzahlGeheiltSum-df_bl.AnzahlTotSum
    return df_bl

ages, gkz, tsbl, tsbez, hosptes, mortvie = update_data()
df_bl = make_df_bl(tsbl, hosptes)
frames = [ages, gkz, tsbl, tsbez, hosptes, mortvie]


# Sieben Tage Inzidenz Fälle
current_date = maxdate = str(df_bl.joindate.max())
indicators = ['SiebenTageInzidenzFaelle','FaelleAktiv' ]
col_list = indicators + ['Bundesland']
df_current_plot = df_bl.loc[(df_bl.joindate==df_bl.joindate.max())][col_list]

fig = px.bar(data_frame=df_current_plot.melt(id_vars='Bundesland', value_vars=indicators),x='value', y='Bundesland',
             template='plotly_white', color_discrete_sequence=px.colors.diverging.Portland_r,  orientation='h', facet_col='variable',
             title=f'7 Tage Inzidenz und Anzahl Aktive Fälle per {current_date}')
fig.layout.xaxis2.update(matches=None)
fig.show()
icu = ['FZICU', 'FZICUFree']+['Bundesland']
df_icu_plot = df_bl.loc[(df_bl.joindate==df_bl.joindate.max())][icu].melt(id_vars='Bundesland', value_vars=['FZICU', 'FZICUFree'])
px.bar(df_icu_plot, x='value', y='Bundesland', color='variable',orientation='h',
       template='plotly_white', color_discrete_sequence=px.colors.diverging.Portland_r, title=f'ICU Belegung und Freie Kapazitäten per {current_date}')
hosp = ['FZHosp', 'FZHospFree']+['Bundesland']
df_icu_plot = df_bl.loc[(df_bl.joindate==df_bl.joindate.max())][hosp].melt(id_vars='Bundesland', value_vars=['FZHosp', 'FZHospFree'])
px.bar(df_icu_plot, x='value', y='Bundesland', color='variable',orientation='h',
       template='plotly_white', color_discrete_sequence=px.colors.diverging.Portland_r,  
       title=f'Hospitalisierung Belegung und Freie Kapazitäten per {current_date}')
bl = df_bl.Bundesland.unique()
columns = df_bl.columns.unique()
start= str(df_bl.joindate.min())
#end = str(df_bl.joindate.max())


@interact_manual
def df_bl_explorer(Bundesland=bl, column=columns, startdate=start):
    startdate = pd.to_datetime(startdate)
    #enddate = pd.to_datetime(enddate)

    display(df_bl.loc[(df_bl.Bundesland==Bundesland) & (df_bl.joindate>=startdate)].set_index('joindate')[['Bundesland']+[column]][0:20])
# Active cases are missing from Dataset - Added Manually
ages['AnzahlAktiv'] = ages.Anzahl-ages.AnzahlGeheilt-ages.AnzahlTot
df_ages_plot = ages.melt(id_vars=['Bundesland','Altersgruppe','Geschlecht'], value_vars=['AnzahlGeheilt', 'AnzahlTot',
       'AnzahlAktiv'])
bl_select = df_ages_plot.Bundesland.unique().tolist()

def show_age_distribution(Bundesland = 'Österreich'):
    fig = px.bar(df_ages_plot.loc[df_ages_plot.Bundesland==Bundesland], y='Altersgruppe', x='value', 
                 color='variable', facet_col='Geschlecht', template='plotly_white', color_discrete_sequence=px.colors.diverging.RdYlBu)
    fig.show()

interact(show_age_distribution,
         Bundesland = widgets.Dropdown(options=bl_select)
        )
plot_vars = ['AnzahlFaelle','AnzahlFaelleSum','FaelleAktiv', 'AnzahlFaelle7Tage', 'SiebenTageInzidenzFaelle','AnzahlTotTaeglich', 
             'AnzahlTotSum', 'AnzahlGeheiltTaeglich','AnzahlGeheiltSum', 'TestGesamt', 'FZHosp', 'FZICU', 'FZHospFree', 'FZICUFree']

mindate = str(df_bl.joindate.min())
maxdate = str(df_bl.joindate.max())


def interactive_timeseries_ages(value_var = plot_vars, start='2020-04-01', end='2020-10-01'):
    
    bl_list=['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 
                          'Tirol', 'Vorarlberg', 'Wien','Österreich']
    try:
        start = pd.to_datetime(start).date()
        end = pd.to_datetime(end).date()

        dfplot = df_bl.loc[(df_bl.joindate>=start) & (df_bl.joindate<=end) & df_bl.Bundesland.isin(list(bl_list))].copy()
        dfplot = dfplot.melt(value_vars=[value_var], id_vars=['joindate','Bundesland'])
        dfplot.loc[dfplot.value==0] = np.nan 
        dfplot.set_index('joindate', inplace=True)

        fig = px.line(dfplot.dropna(), y='value', color='Bundesland', title=value_var, template='plotly_white',
                     color_discrete_sequence=px.colors.qualitative.Bold_r)
        fig.show()
    except:
        print('Invalid Selection')


interact(interactive_timeseries_ages,
                value_var = widgets.Dropdown(options=plot_vars),
                start=widgets.DatePicker(value=pd.to_datetime(mindate)),
                end=widgets.DatePicker(value=pd.to_datetime(maxdate))
               )
austria = gpd.read_file('../input/austria-districts-shapefile/STATISTIK_AUSTRIA_POLBEZ_20200101Polygon.shp')

austria.head()
austria.id = austria.id.astype(int)
tsbez_geo = tsbez.merge(austria, left_on='GKZ', right_on='id', how='outer' )
tsbez_geo['Date'] = pd.to_datetime(tsbez_geo.Time, dayfirst=True).dt.date


# taking only last date into geoDF 
lastdate = tsbez_geo.Date.dropna().max()
geo = gpd.GeoDataFrame(tsbez_geo.loc[tsbez_geo.Date==lastdate].dropna())
geo.info()
indicators = ['SiebenTageInzidenzFaelle','AnzEinwohner', 'AnzahlFaelle','AnzahlFaelleSum', 'AnzahlFaelle7Tage','AnzahlTotTaeglich', 
              'AnzahlTotSum', 'AnzahlGeheiltTaeglich','AnzahlGeheiltSum']


nr_cat = 8

@interact
def showmap_per_category(indicator=indicators):
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    #divider = make_axes_locatable(ax)
    tmp = geo.copy()
    #cax = divider.append_axes("right", size="3%", pad=-1) #resize the colorbar
    tmp[indicator+'_cat'] = pd.cut(tmp[indicator], bins=nr_cat)

    tmp.plot(column=indicator+'_cat', ax=ax,  legend=True, cmap='Reds')
    plt.title(indicator)
    fig = tmp.geometry.boundary.plot(color='#505050', ax=ax, linewidth=0.3) #Add some borders to the geometries
    ax.axis('off')
    plt.show(fig)

# widgets for data update - not working yet properly
update_data_btn = widgets.Button(description="Load Data")
last_date_lbl = widgets.Label(value='na')

#get last available data to show on label
def get_last_date():
    return str(df_bl.joindate.iloc[-1])
def update_label(lbl):
    lbl.value = "update"
    lbl.update()
    
# corresponding event handlers
def update_data_btn_event(button):
    #update_data()
    make_df_bl()
    update_label(last_date_lbl)
disp(update_data_btn)
disp(last_date_lbl)
df_bl.head()
# check if all dataframes were loaded correctly
[display(f.head()) for f in frames]

def make_ts_plot(value_var='AnzahlFaelleSum', start_date='2020-04-01', end_date='2020-10-10',  
                 bl_list=['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 
                          'Tirol', 'Vorarlberg', 'Wien','Österreich'], title='Anzahl Fälle'):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    dfplot = df_bl.loc[(df_bl.joindate>=start_date) & (df_bl.joindate<=end_date) & df_bl.Bundesland.isin(bl_list)].copy()
    dfplot = dfplot.melt(value_vars=[value_var], id_vars=['joindate','Bundesland'])
    dfplot.loc[dfplot.value==0] = np.nan 
    dfplot.set_index('joindate', inplace=True)
    fig = px.line(dfplot.dropna(), y='value', color='Bundesland', title=title)
    return fig


bundesland_list = ['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich',
       'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg', 'Wien',
       'Österreich']

f = make_ts_plot(value_var='TestGesamt', title='FZICUFree', bl_list=['Wien', 'Oberösterreich'])
f.show()
# Widgets for creation of graph
plot_vars = ['AnzahlFaelle','AnzahlFaelleSum', 'AnzahlFaelle7Tage', 'SiebenTageInzidenzFaelle','AnzahlTotTaeglich', 
             'AnzahlTotSum', 'AnzahlGeheiltTaeglich','AnzahlGeheiltSum', 'TestGesamt', 'FZHosp', 'FZICU', 'FZHospFree', 'FZICUFree']

import plotly.graph_objs as go

var = widgets.Dropdown(options=plot_vars)
bl_select = widgets.SelectMultiple(options=df_bl.Bundesland.unique(),
                                  value=tuple(df_bl.Bundesland.unique()))                                   
update_graph_btn = widgets.Button(description='Update Graph')
interact_graph = go.FigureWidget()

import ipywidgets as widgets
import pandas as pd

start_date = df_bl.joindate.min()
end_date = df_bl.joindate.max()

dates = pd.date_range(start_date, end_date, freq='D')

options = [(date.strftime(' %d %m %Y '), date) for date in dates]
index = (0, len(options)-1)

selection_range_slider = widgets.SelectionRangeSlider(
    options=options,
    index=index,
    description='Dates',
    orientation='horizontal',
    layout={'width': '500px'}
)


def update_graph(obj):
    print(var.value)
    print(bl_select.value)
    print(selection_range_slider.value)
    interact_graph = make_ts_plot(value_var='TestGesamt', title='FZICUFree', bl_list=['Wien','Niederösterreich', 'Oberösterreich'])
    f.show()

update_graph_btn.on_click(update_graph)

allwid = widgets.VBox(children=[var, bl_select, selection_range_slider,update_graph_btn, interact_graph])
disp(allwid)
# Manual first try for plots
dfplot = df_bl.melt(value_vars=['SiebenTageInzidenzFaelle'], id_vars=['joindate','Bundesland'])
dfplot.set_index('joindate', inplace=True)
px.line(dfplot.dropna(), y='value', color='Bundesland', title='Sieben TageInzidenz Faelle')



myslider = widgets.IntSlider(
    min=0,
    max=10,
    step=1,
    description='Slider:',
    value=3
)
disp(myslider)
btn = widgets.Button(description='Medium')
disp(btn)
def btn_eventhandler(obj):
    print('Hello from the {} button!'.format(obj.description))
btn.on_click(btn_eventhandler)

# all available widgets
print(dir(widgets))
bl_dropdown = widgets.Dropdown(options = df_bl.Bundesland.unique().tolist())
def bl_dropdown_event(change):
    #print(f'Value set to {change.new}')
    print(f'Value changed to {change.new}')
bl_dropdown.observe(bl_dropdown_event, names='value')

disp(bl_dropdown)
print(bl_dropdown.value)

hosptes.columns
hosptes.loc[hosptes.BundeslandID==9]['TestGesamt'].plot()
pd.to_datetime(hosptes.Meldedat, dayfirst=True)
tuple(df_bl.Bundesland.unique())
from ipywidgets import interact, interact_manual
@interact
def show_more_cases_than(column='AnzahlFaelle7Tage', x=200):
    return df_bl.loc[df_bl[column]>x]
# with selected columns
@interact
def get_highest_values(column=df_bl.columns.to_list()):
    return(df_bl.loc[df_bl[column] == df_bl[column].max()])

def interactive_plot_seaborn(value_var, start, end):
    
    #not interactive yet
    
    bl_list=['Burgenland', 'Kärnten', 'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 
                          'Tirol', 'Vorarlberg', 'Wien','Österreich']
    
    #start = start.date()
    #end = end.date()
    
    dfplot = df_bl.loc[(df_bl.joindate>=start) & (df_bl.joindate<=end) & df_bl.Bundesland.isin(list(bl_list))].copy()
    dfplot = dfplot.melt(value_vars=[value_var], id_vars=['joindate','Bundesland'])
    dfplot.loc[dfplot.value==0] = np.nan 
    dfplot.set_index('joindate', inplace=True)
    plt.figure(figsize=(15,10))
    sns.set_palette('colorblind')
    sns.set_style('whitegrid')
    
    sns.lineplot(data=dfplot.dropna(), x=dfplot.dropna().index, y='value', hue='Bundesland', )
    plt.show()
    #fig = px.line(dfplot.dropna(), y='value', color='Bundesland', title=title)
    #fig.show()
interactive_plot_seaborn('AnzahlFaelle', pd.to_datetime('2020-01-01'),pd.to_datetime('2020-12-31'))
interact_manual(interactive_plot_seaborn, 
         start=widgets.DatePicker(value=pd.to_datetime('2020-01-01')),
         end=widgets.DatePicker(value=pd.to_datetime('2020-12-31')),
         value_var = widgets.Dropdown(options=plot_vars),)
import pandas as pd
import datetime as dt
dt.date.today()-dt.date(2011,8,8)
365*9+3
print('Test')