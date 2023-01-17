import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None

df = pd.read_csv('/kaggle/input/bart-ridership/ridership/date-hour-soo-dest-2019.csv')

df.head()
df.info()
df['Origin Station'].unique()
plt.figure(figsize=(20,20))
img = plt.imread('/kaggle/input/bart-map/BART System Map API.png')
plt.imshow(img)
plt.show()
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import base64

def animate_stations(df, x_col, y_col, animation_frame, size, hover_name, title):

    init_notebook_mode()
    fig = px.scatter(df, 
                 x=x_col,
                 y=y_col, 
                 animation_frame= animation_frame, 
                 size= size, 
                 hover_name = hover_name,
                 range_x=(0,2050), 
                 range_y=(2050,0), 
                 width=700, 
                 height=700,
                 labels = {'origin_x':'', 'origin_y':''})
    image_filename = "/kaggle/input/bart-map-black-and-white/BART System Map API BW.png"
    plotly_logo = base64.b64encode(open(image_filename, 'rb').read())
    fig.update_layout(xaxis_showgrid=False, 
                    yaxis_showgrid=False,
                    xaxis_showticklabels=False,
                    yaxis_showticklabels=False,
                    title= title,
                    images= [dict(
                    source='data:image/png;base64,{}'.format(plotly_logo.decode()),
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    xanchor="left",
                    yanchor="top",
                    sizing="stretch",
                    layer="below")])
    iplot(fig)
station_coords = {
    '12TH':[798, 702],
    '19TH':[798, 655],
    'MCAR':[802, 607],
    '16TH':[375, 948],
    'CIVC':[397, 915],
    'POWL':[419, 886],
    'MONT':[438, 860],
    'EMBR':[457, 834],
    '24TH':[358, 988],
    'GLEN':[339, 1023],
    'BALB':[318, 1052],
    'DALY':[288, 1100],
    'COLM':[301, 1154],
    'SSAN':[342, 1200],
    'SBRN':[380, 1270],
    'SFIA':[485, 1321],
    'MLBR':[495, 1435],
    'WOAK':[679, 793],
    'LAKE':[846, 871],
    'FTVL':[903, 928],
    'COLS':[959, 983],
    'SANL':[1025, 1045],
    'BAYF':[1075, 1105],
    'HAYW':[1196, 1245],
    'SHAY':[1280, 1335],
    'UCTY':[1360, 1420],
    'FRMT':[1455, 1525],
    'WARM':[1505, 1635],
    'MLPT':[1525, 1765],
    'BERY':[1535, 1895],
    'ROCK':[886, 542],
    'ORIN':[965, 492],
    'LAFY':[1055, 444],
    'WCRK':[1135, 394],
    'PHIL':[1225, 337],
    'CONC':[1295, 294],
    'NCON':[1375, 244],
    'PITT':[1535, 182],
    'PCTR':[1695, 193],
    'ANTC':[1855, 187],
    'CAST':[1265, 1125],
    'WDUB':[1505, 1105],
    'DUBL':[1655, 1085],
    'OAKL':[884, 1065],
    'ASHB':[765, 537],
    'DBRK':[735, 492],
    'NBRK':[706, 450],
    'PLZA':[673, 407],
    'DELN':[644, 356],
    'RICH':[611, 310]
}
    
def get_x_coord(station):
    return station_coords[station][0]

def get_y_coord(station):
    return station_coords[station][1]

df_day = df[df.Date == '2019-01-09']
df_day['origin_x'] = df_day['Origin Station'].apply(lambda x: get_x_coord(x))
df_day['origin_y'] = df_day['Origin Station'].apply(lambda x: get_y_coord(x))
df_day['destination_x'] = df_day['Destination Station'].apply(lambda x: get_x_coord(x))
df_day['destination_y'] = df_day['Destination Station'].apply(lambda x: get_y_coord(x))
df_day['Route'] = df_day[['Origin Station', 'Destination Station']].apply(lambda x: '-'.join(x), axis=1)

df_day_origin = df_day.groupby(['Hour', 'Origin Station', 'origin_x', 'origin_y']).sum().reset_index()
animate_stations(df_day_origin, 'origin_x', 'origin_y', 'Hour', 'Trip Count', 'Origin Station', 'Departures: 2019-01-09')

df_day_destination = df_day.groupby(['Hour', 'Destination Station', 'destination_x', 'destination_y']).sum().reset_index()
animate_stations(df_day_destination, 'destination_x', 'destination_y', 'Hour', 'Trip Count', 'Destination Station', 'Arrivals: 2019-01-09')

fig = px.bar(df_day.groupby(['Hour']).sum().reset_index(), x='Hour', y='Trip Count', hover_name='Trip Count', title='Total Trips each Hour 1/9/2019')
fig.show()

idx = df_day.groupby(['Hour'])['Trip Count'].transform(max) == df_day['Trip Count']
fig = px.bar(df_day[idx], x='Hour', y='Trip Count', hover_name = 'Route', color='Route', title = 'Most Travelled Routes each Hour 1/9/2019')
fig.show()

df_day_o = df_day.groupby(['Hour', 'Origin Station']).sum().reset_index()
idx = df_day_o.groupby(['Hour'])['Trip Count'].transform(max) == df_day_o['Trip Count']
fig = px.bar(df_day_o[idx], x='Hour', y='Trip Count', hover_name = 'Origin Station', color='Origin Station', title='Most common Origin Station each Hour 1/9/2019')
fig.show()

df_day_d = df_day.groupby(['Hour', 'Destination Station']).sum().reset_index()
idx = df_day_d.groupby(['Hour'])['Trip Count'].transform(max) == df_day_d['Trip Count']
fig = px.bar(df_day_d[idx], x='Hour', y='Trip Count', hover_name = 'Destination Station', color='Destination Station', title='Most common Destination Station each Hour 1/9/2019')
fig.show()

MAX_SHOW = 40
df_day['Hour_string'] = df_day['Hour'].astype(str)
df_day['Route:Hour'] = df_day[['Route', 'Hour_string']].apply(lambda x: ':'.join(x), axis=1)

fig = px.bar(df_day.sort_values(by=['Trip Count'], ascending=False)[0:MAX_SHOW].reset_index(), x='Route:Hour', y='Trip Count', hover_name='Route:Hour', color='Hour', title='Most Common Route:Hour Combinations 1/9/2019')
fig.show()

fig = px.bar(df_day.groupby(['Origin Station']).sum().sort_values(by = ['Trip Count'], ascending=False).reset_index(), x = 'Origin Station', y = 'Trip Count', hover_name = 'Origin Station', title='Most Used Stations (Departures) 1/9/2019')
fig.show()

fig = px.bar(df_day.groupby(['Destination Station']).sum().sort_values(by = ['Trip Count'], ascending=False).reset_index(), x = 'Destination Station', y = 'Trip Count', hover_name = 'Destination Station', title='Most Used Stations (Arrivals) 1/9/2019')
fig.show()

df_day_o = df_day.groupby(['Origin Station']).sum().sort_values(by = ['Trip Count'], ascending=False).reset_index()
df_day_d = df_day.groupby(['Destination Station']).sum().sort_values(by = ['Trip Count'], ascending=False).reset_index()
df_day_o.rename(columns={'Origin Station': 'Station'}, inplace=True)
df_day_d.rename(columns={'Destination Station': 'Station'}, inplace=True)
df_day_o['Trip Type'] = 'Departure'
df_day_d['Trip Type'] = 'Arrival'
df_day_t = pd.concat([df_day_o, df_day_d]).reset_index()
fig = px.bar(df_day_t.sort_values(by = ['Trip Count'], ascending=False).reset_index(), x = 'Station', y = 'Trip Count', hover_name = 'Station', color='Trip Type', title='Most Used Stations (Departures and Arrivals) 1/9/2019')
fig.show()


def show_station(station_name, df_day, plot_title):
    df_o = df_day[df_day["Origin Station"] == station_name].groupby(['Hour']).sum().reset_index()
    df_d = df_day[df_day["Destination Station"] == station_name].groupby(['Hour']).sum().reset_index()
    df_o["Trip Type"] = 'Departure'
    df_d["Trip Type"] = 'Arrival'
    df = pd.concat([df_o, df_d]).reset_index()
    fig = px.bar(df, x='Hour', y='Trip Count', color='Trip Type', hover_name='Trip Type', title=plot_title)
    fig.show()

show_station('MONT', df_day, 'MONT (Montgomery St.) Station Usage 1/9/2019')
show_station('EMBR', df_day, 'EMBR (Embarcadero) Station Usage 1/9/2019')
show_station('RICH', df_day, 'RICH (Richmond) Station Usage 1/9/2019')
show_station('DUBL', df_day, 'DUBL (Dublin/Pleasanton) Station Usage 1/9/2019')
def category_score(station_name, df_day):
    df_o = df_day[df_day["Origin Station"] == station_name].groupby(['Hour']).sum().reset_index()
    df_d = df_day[df_day["Destination Station"] == station_name].groupby(['Hour']).sum().reset_index()
    if len(df_d) == 0 or len(df_o) == 0:
        return 0
    else:
        waah = np.average(df_d['Hour'],weights=df_d['Trip Count'])
        wadh = np.average(df_o['Hour'],weights=df_o['Trip Count'])
    return waah - wadh

def category_score_df(station_coords, df_day):
    stations = list(station_coords.keys())
    category_scores = []
    x_coords = []
    y_coords = []
    for station in stations:
        category_scores.append(category_score(station, df_day))
        loc = station_coords[station]
        x_coords.append(loc[0])
        y_coords.append(loc[1])
    df = pd.DataFrame.from_dict({'station':stations, 'category_score':category_scores, 'x_coord':x_coords, 'y_coord':y_coords})
    df = df.loc[df['category_score'] != 0]
    df['category_score'] = df['category_score'] * (1/df['category_score'].abs().max())
    return df
    
df_cat = category_score_df(station_coords, df_day)

init_notebook_mode()
fig = px.scatter(df_cat, 
                 x='x_coord',
                 y='y_coord',  
                 hover_name = 'station',
                 color = 'category_score',
                 range_x=(0,2050), 
                 range_y=(2050,0), 
                 width=700, 
                 height=700,
                 labels = {'origin_x':'', 'origin_y':''})
image_filename = "/kaggle/input/bart-map-black-and-white/BART System Map API BW.png"
plotly_logo = base64.b64encode(open(image_filename, 'rb').read())
fig.update_layout(xaxis_showgrid=False, 
                    yaxis_showgrid=False,
                    xaxis_showticklabels=False,
                    yaxis_showticklabels=False,
                    title= 'Station by Category Score 1/9/2019',
                    images= [dict(
                    source='data:image/png;base64,{}'.format(plotly_logo.decode()),
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    xanchor="left",
                    yanchor="top",
                    sizing="stretch",
                    layer="below")])
iplot(fig)

fig = px.bar(df_cat.sort_values(['category_score']), x='station', y='category_score', color='category_score', hover_name='station', title='Station by Category Score 1/9/2019')
fig.show()

show_station('OAKL', df_day, 'OAKL (Oakland International Airport) Station Usage 1/9/2019')
df_fare = pd.read_csv('/kaggle/input/bart-fares/BART_fares.csv')
df_fare
#import matplotlib.pyplot as plt 
#import mpld3 from mpld3 
#import plugins 
#img = plt.imread("/kaggle/input/bart-map/BART System Map API.png") 
#fig, ax = plt.subplots(figsize=(14,14)) ax.imshow(img) plt.grid('on')
#plt.axis('off') plt.scatter([457], [834]) plugins.connect(fig, plugins.MousePosition(fontsize=14))
#plt.show() mpld3.enable_notebook()