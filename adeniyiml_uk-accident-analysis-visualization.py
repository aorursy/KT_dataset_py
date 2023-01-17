import math 

import calendar

import pandas as pd

import datetime





import geopandas as gpd

from shapely.geometry import Point

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from plotly.subplots import make_subplots

import plotly.graph_objects as go

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# plotly

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.express as px
#load in datasets

uk_2007 = pd.read_csv('../input/uk-acc-2005-2007/accidents_2005_to_2007.csv', low_memory=False)

uk_2011 = pd.read_csv('../input/uk-accident-datasets/accidents_2009_to_2011.csv', low_memory=False)

uk_2014 = pd.read_csv('../input/uk-accident-datasets/accidents_2012_to_2014.csv', low_memory=False)
uk_accidents = pd.concat([uk_2007, uk_2011, uk_2014], ignore_index=True)
uk_accidents.shape
#Number of nulls by each columns

uk_accidents.isnull().sum()
#drop columns with greater than 10,000 null values

uk_accidents = uk_accidents.drop(['Junction_Detail', 'Junction_Control', 'LSOA_of_Accident_Location'], axis=1)
uk_accidents.dropna(inplace = True)
uk_accidents.shape
uk_accidents.head(5)
#load additional datasets

uk_LAD = pd.read_csv('../input/uk-accident-datasets/LAD.csv')

uk_LAD3 = pd.read_csv('../input/uk-accident-datasets/LAD3.csv')

uk_LAD4 = pd.read_csv('../input/uk-accident-datasets/LAD4.csv')
uk_LAD.head()
uk_LAD3.head(5)
#Road Network Classification

uk_LAD4.head(10)
#lets merge the two dataframe based on the common column that both dataframe have which is 'Sub-Group'



uk_LAD2 = pd.merge(uk_LAD3, uk_LAD4, on='Sub-Group')
#columns with more than 100,000 None Values

none_columns = []

for x in uk_accidents.columns:

    none_ct = uk_accidents[x].loc[uk_accidents[x] == 'None'].count()

    if none_ct >= 100000:

        none_columns.append(x)
print(none_columns)
uk_accidents = uk_accidents.drop(none_columns, axis=1)
uk_accidents.shape
#rename ONS to LAD_Code

uk_accidents.rename(columns={'Local_Authority_(Highway)': 'LAD_Code'}, inplace = True)

uk_LAD2.rename(columns={'ONS Code': 'LAD_Code'}, inplace = True)

uk_LAD.rename(columns={'Code': 'LAD_Code'}, inplace = True)
uk_LAD2 = pd.merge(uk_LAD, uk_LAD2, on='LAD_Code')
uk_accidents = pd.merge(uk_accidents, uk_LAD2, on='LAD_Code')
df_uk_gpd = uk_accidents.copy()
#due to the large size of the accident dataset (~1.5 million rows)

#i will only convert the accident df for year 2014 to "GeoDataFrame"

df_uk_2014 = df_uk_gpd[df_uk_gpd['Year'] == 2014]
df_uk_gpd_2014 = df_uk_2014.copy()
#convert accident df

points = df_uk_gpd_2014.apply(lambda row: Point(row.Location_Easting_OSGR, row.Location_Northing_OSGR), axis=1)

df_uk_gpd_2014 = gpd.GeoDataFrame(df_uk_gpd_2014, geometry=points)
#uk 2018 road network map

gb_shape = gpd.read_file('../input/shape-file/2018-MRDB-minimal.shp')
gb_shape.shape
#lets plot points where road accidents occured based on various columns/aspect of the data

def map_plot(df1, df2, column, column_title, color):

    

    ax = df1.plot(figsize=(30,15), color='black', linewidth=0.6)

    df2.plot(column= column, ax=ax, markersize=60, legend = True, cmap=color, edgecolor='white')

    leg = ax.get_legend()

    leg.set_bbox_to_anchor((1.1,0.5))

    leg.set_title(column_title)

    leg.set_frame_on(False)

    ax.set_axis_off()

    ax.set_title('Locations of Road Accidents in UK based on the ' + column_title + ', 2014', fontsize=16, pad=10)
map_plot(gb_shape, df_uk_gpd_2014, 'Region/Country', 'Regions', "gist_rainbow_r")
map_plot(gb_shape, df_uk_gpd_2014, 'Supergroup name', 'Settlement Areas', "plasma_r")
map_plot(gb_shape, df_uk_gpd_2014, 'Sub-Group Description', "Road Networks", "rainbow")
sns.set_context('talk')

sns.set_style("darkgrid")

g = sns.FacetGrid(df_uk_gpd, col="Year", hue="Region/Country", palette="gist_rainbow_r", col_wrap=3, height=6)

g.map(plt.scatter, "Longitude", "Latitude")

g.add_legend()

g.fig.subplots_adjust(top=0.93)

plt.suptitle('Yearwise trend of road accidents in UK between 2005 - 2014 for different regions', fontsize=24)

plt.show()
#convert all datetime columns to datetime formats

df_uk = uk_accidents.copy()

df_uk['Date'] = pd.to_datetime(df_uk['Date'])

df_uk['Year'] = df_uk['Date'].dt.year
df_uk['Day'] = df_uk.Date.dt.day

df_uk['week_in_month'] = pd.to_numeric(df_uk.Day/7)

df_uk['week_in_month'] = df_uk['week_in_month'].apply(lambda x: math.ceil(x))

df_uk['month'] = df_uk.Date.dt.month
#datetime.time(df_uk['Time'])

df_uk['Time'] = pd.to_timedelta(df_uk['Time'] +':00')
df_uk['Time']
df_uk['Hour'] = df_uk['Time'].dt.components['hours']
df_uk.Hour.unique()
def groupby_accidents(df, column):

    col_agg = df.groupby(column).Number_of_Casualties.agg(['sum', 'count', 'mean'])

    col_agg.reset_index(inplace = True)

    col_agg.sort_values(by = column, inplace = True)

    return col_agg
year_agg = groupby_accidents(df_uk, 'Year')
data = go.Scatter(x = year_agg['Year'], y=year_agg['sum'], mode="lines+markers", name='Number of Casualties', 

                  line= dict(color = ('rgb(255,165,0)'), width=4), showlegend = True)

layout = go.Layout(title='<b> Records of Road Accidents (Casualties per year) in UK between 2005 - 2014 <b>',

                   xaxis=dict(title='<b> Years <b>',titlefont=dict(size=16, color='#7f7f7f')),

                   yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'))

                 )

fig = go.Figure(data=data, layout = layout)

fig.update_xaxes(dtick=1)



iplot(fig)
#records of road accidents based on different regions

def plot_regions_agg(df, regions, content):

    

    def groupby_region(df, column, name):

        

        col_agg = df[df['Region/Country'] == name].groupby(column).Number_of_Casualties.agg(['count', 'sum'])

        col_agg.reset_index(inplace = True)

        col_agg.sort_values(by = column, inplace = True)

        return col_agg

    

    region_traces = []

    colors = ['lightslategray', 'crimson', 'darkcyan', 'darkgoldenrod', 'cornsilk', 'turquoise', 'limegreen', \

              'darkorchid', 'palevioletred', 'forestgreen', 'silver', 'lightsteelblue']

    for name in range(len(regions)):

        name_agg = groupby_region(df, 'Year', regions[name])

        data_agg = go.Scatter(x = name_agg['Year'], y= name_agg['sum'], mode="lines+markers", name=regions[name], 

                              line= dict(color = colors[name], width=2.5))

        region_traces.append(data_agg)

    layout = go.Layout(title='<b> Rates of Road Accidents in different ' + content + ' regions between 2005 - 2014<b>', width=1100, 

                       height=600, xaxis=dict(title='<b> Year <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 

                       yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')))

        

    fig = go.Figure(data=region_traces, layout = layout)

    fig.update_xaxes(dtick=1)

    fig.show()
regions = df_uk['Region/Country'].unique().tolist()

plot_regions_agg(df_uk, regions, 'UK')
areas = ['London, England', 'North East, England', 'Yorkshire and The Humber, England', 'North West, England']

plot_regions_agg(df_uk, areas, 'England')
region_agg = groupby_accidents(df_uk, 'Region/Country').sort_values(by = 'sum', ascending = False)

fig = px.bar(region_agg, x= 'Region/Country', y= 'sum', color='sum', 

             labels={'sum':'<b> Number of Casualties <b>'}, width=1000, height=700,

             color_continuous_scale=px.colors.sequential.Plasma)



fig.update_layout(title='<b> Road Accidents in UK (based on different regions) between 2005 - 2014 <b>', xaxis_title='<b> Year <b>',

                  yaxis_title='<b> Number of Casualties <b>')

fig.show()
top_20 = groupby_accidents(df_uk, 'Highway Authority').sort_values(by = 'sum', ascending = False)[:20]

bottom_20 = groupby_accidents(df_uk, 'Highway Authority').sort_values(by = 'sum', ascending = False)[-20:]
#Data Visualization

def plot_highway(df, name, color):

    fig = px.bar(df, x= 'Highway Authority', y= 'sum', color='sum', 

                 labels={'sum':'<b> Casualties <b>'}, width=1000, height=700,

                 color_continuous_scale=color)

    

    if name == 'Safest':

        content = 'lowest'

    else:

        content = 'highest'

    

    fig.update_layout(title='<b> 20 ' + name + ' Highway Authorities with the ' + content + ' record of road accidents in UK <b>', xaxis_title='<b> Highway Authority <b>', 

                      yaxis_title='<b> Number of Casualties <b>')

    fig.show()
plot_highway(top_20, 'Dangerous', px.colors.sequential.Cividis)
plot_highway(bottom_20, 'Safest', px.colors.sequential.Inferno)
def plot_time_agg(df, column):

    

    def groupby_col(df, column, year):

        

        col_agg = df[df['Year'] == year].groupby(column).Number_of_Casualties.agg(['count', 'sum'])

        if year == None:

            col_agg = df.groupby(column).Number_of_Casualties.agg(['count', 'sum'])

            col_agg['average'] = col_agg['sum'] / 9

        col_agg.reset_index(inplace = True)

        col_agg.sort_values(by = column, inplace = True)

        if column == 'month':

            col_agg['month'] = col_agg['month'].apply(lambda x: calendar.month_abbr[x])

        if column == 'Day_of_Week':

            col_agg['Day_of_Week'] = col_agg['Day_of_Week'] - 1

            col_agg['Day_of_Week'] = col_agg['Day_of_Week'].apply(lambda x: calendar.day_abbr[x])

            

        return col_agg

    

    year_list = df_uk['Year'].unique().tolist() + [None]

    list_of_traces = []

    colors = ['darkmagenta', 'deeppink', 'lavender', 'lightsteelblue', 'orchid', 'navy', 'forestgreen', \

              'greenyellow', 'silver', 'darkslategrey']

    

    for year in range(len(year_list)):

        if year_list[year] == None:

            name_agg = groupby_col(df, column, year_list[year])

            data_agg = go.Scatter(x = name_agg[column], y= name_agg['average'], mode="lines+markers", name='Overall Average', 

                                 line= dict(color = 'darkslategrey', width=4))

        else:

            name_agg = groupby_col(df, column, year_list[year])

            data_agg = go.Scatter(x = name_agg[column], y= name_agg['sum'], mode="lines+markers", name=regions[year], 

                                  line= dict(color = colors[year], width=2, dash = 'dashdot'))

        list_of_traces.append(data_agg)

        

        

    #Data visualization

    if column == 'month':

        content = 'by Months of the year'

        tk = ''

    elif column == 'Hour':

        content = 'during Day and Night'

        tk = 1

    elif column == 'Day_of_Week':

        content = 'by weekdays'

        tk = ''

    elif column == 'week_in_month':

        content = 'by weeks in a month'

        tk = ''

    

    layout = go.Layout(title='<b> Occurences of Road Accidents ' + content + ' in UK between 2005 - 2014<b>', 

                       xaxis=dict(title='<b> ' + column + ' <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 

                       yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')))

    

    fig = go.Figure(data=list_of_traces, layout = layout)

    fig.update_xaxes(dtick=tk)

    iplot(fig)
plot_time_agg(df_uk, 'Day_of_Week')
plot_time_agg(df_uk, 'Hour')
plot_time_agg(df_uk, 'week_in_month')
plot_time_agg(df_uk, 'month')
network_agg = groupby_accidents(df_uk, 'Sub-Group Description').sort_values(by = 'sum', ascending = False)
network_agg
#Data Visualization



def visualize_aggregates(df, column, color):

    

    fig = px.bar(df, x= column, y= 'sum', color='sum', 

                 labels={'sum':'<b> Casualties <b>'}, width=1000, height=700, 

                 color_continuous_scale=color)

    

    if column == 'Sub-Group Description':

        content = 'Road Networks in UK with the highest record of road accidents, 2005 - 2014 '

    elif column == 'Road_Type':

        content = 'Road Types in UK with the highest record of road accidents, 2005 - 2014'

    elif column == 'Conditions':

        content = 'Weather, Road & Light Conditions at the time of Road accidents in UK between 2005 - 2014'

    elif column == 'Pedestrian_Crossing':

        content = 'Pedestrian Crossings at the locations of Road accidents in UK between 2005 - 2014'

    elif column == 'Speed_limit':

        content = 'Speed Limits associated with Road accidents in UK between 2005 - 2014 '

    

    fig.update_layout(title='<b> ' + content + ' <b>', 

                      xaxis=dict(title='<b> ' + column + ' <b>',titlefont=dict(size=16), tickfont=dict(size=13, color='darkslateblue')), 

                      yaxis=dict(title='<b> Number of Casualties <b>',titlefont=dict(size=16), tickfont=dict(size=13, color='darkslateblue'))) 

    

    fig.show()
visualize_aggregates(network_agg, 'Sub-Group Description', px.colors.diverging.Spectral)
road_agg = groupby_accidents(df_uk, 'Road_Type').sort_values(by = 'sum', ascending = False)
visualize_aggregates(road_agg, 'Road_Type', px.colors.diverging.RdBu)
#lets look at the distinct condition underwhich road accidents occured



conditions_columns = ['Road_Surface_Conditions', 'Weather_Conditions', 'Light_Conditions']

print('Conditions under which road accidents occured: \n')

for column in conditions_columns:

    print(column + ': ')

    print(df_uk[column].unique().tolist())

    print('\n')
roadsurface_agg = groupby_accidents(df_uk, 'Road_Surface_Conditions')

weather_agg = groupby_accidents(df_uk, 'Weather_Conditions')

light_agg = groupby_accidents(df_uk, 'Light_Conditions')
#convert 'other' value in weather_agg df to 'Unknown'

weather_agg['Weather_Conditions'] = weather_agg['Weather_Conditions'].replace({'Other': "Unknown"})
#rename all the conditions to 'Conditions'

roadsurface_agg.rename(columns={'Road_Surface_Conditions': 'Conditions'}, inplace = True)

weather_agg.rename(columns={'Weather_Conditions': 'Conditions'}, inplace = True)

light_agg.rename(columns={'Light_Conditions': 'Conditions'}, inplace = True)
condition_agg = pd.concat([roadsurface_agg, weather_agg, light_agg], ignore_index=True).sort_values(by = 'sum', ascending = False)
condition_agg
visualize_aggregates(condition_agg, 'Conditions', px.colors.diverging.RdYlGn)
#we have two pedestrian crossing columns, lets check them out

#lets look at the distinct condition underwhich road accidents occured



pedestrian_columns = ['Pedestrian_Crossing-Physical_Facilities', 'Pedestrian_Crossing-Human_Control']

print('Pedestrian crossing at the time of road accidents: \n')

for column in pedestrian_columns:

    print(column + ': ')

    print(df_uk[column].unique().tolist())

    print('\n')
pedcrs1_agg = groupby_accidents(df_uk, 'Pedestrian_Crossing-Physical_Facilities')

pedcrs2_agg = groupby_accidents(df_uk, 'Pedestrian_Crossing-Human_Control')
pedcrs1_agg.rename(columns={'Pedestrian_Crossing-Physical_Facilities': 'Pedestrian_Crossing'}, inplace = True)

pedcrs2_agg.rename(columns={'Pedestrian_Crossing-Human_Control': 'Pedestrian_Crossing'}, inplace = True)
pedestrian_agg = pd.concat([pedcrs1_agg, pedcrs2_agg], ignore_index=True).sort_values(by = 'sum', ascending = False)
pedestrian_agg
visualize_aggregates(pedestrian_agg, 'Pedestrian_Crossing', px.colors.diverging.Spectral)
#visualize the distribution

def visualize_distribution(df, column, color):

    

    fig = px.histogram(df, x=column, color_discrete_sequence = color, 

                   width = 1000, height = 700)

    

    if column == 'Number_of_Casualties':

        content = 'Frequency of Casualties due to Road Accidents in UK between 2005 - 2014'

    elif column == 'Number_of_Vehicles':

        content = 'Distribution of Vehicles involved in Road Accidents in UK between 2005 - 2014'

    

    fig.update_layout(title='<b>' + content + '<b>', 

                      xaxis=dict(range=[0, 20], title='<b> ' + column + ' <b>',titlefont=dict(size=16, color='#7f7f7f'), 

                                 tickfont=dict(size=15, color='darkslateblue')), 

                      yaxis=dict(title='<b> Frequency <b>',titlefont=dict(size=16,color='#7f7f7f'), 

                                 tickfont=dict(size=15, color='darkslateblue')))

    

    fig.update_xaxes(dtick=1)

    fig.show()
visualize_distribution(df_uk, 'Number_of_Casualties', px.colors.diverging.balance)
visualize_distribution(df_uk, 'Number_of_Vehicles', px.colors.diverging.RdYlBu)
speed_agg = groupby_accidents(df_uk, 'Speed_limit').sort_values(by = 'sum', ascending = False)
visualize_aggregates(speed_agg, 'Speed_limit', px.colors.diverging.Portland)
severity_agg = groupby_accidents(df_uk, 'Accident_Severity').sort_values(by = 'sum')
severity_agg["Accident_Severity"].replace({1: "Fatal", 2: "Serious", 3: "Slight"}, inplace=True)
area_agg = groupby_accidents(df_uk, 'Urban_or_Rural_Area').sort_values(by = 'sum')
area_agg["Urban_or_Rural_Area"].replace({1: "Urban", 2: "Rural", 3: "Unallocated"}, inplace=True)
### GET THE LABELS AND VALUES FOR THE PIE CHART ###

labels1 = severity_agg["Accident_Severity"].values.tolist()

labels2 = area_agg["Urban_or_Rural_Area"].values.tolist()

values_acc = severity_agg["sum"].values.tolist()

values_area = area_agg['sum'].values.tolist()
# Create subplots, using 'domain' type for pie charts



night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)', 'rgb(6, 4, 4)']

cafe_colors =  ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)', 'rgb(35, 36, 21)']



specs = [[{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=1, cols=2, specs=specs, subplot_titles=['<b> Percent of Road Accidents Severity <b>', 

                                                                 '<b> Percent of Areas affected by Road Accidents <b>'])



# Define pie charts

fig.add_trace(go.Pie(labels=labels1, values=values_acc, name='Accident Severity',

                     marker_colors= night_colors, textinfo='label+percent', insidetextorientation='tangential'), 1, 1)

fig.add_trace(go.Pie(labels=labels2, values=values_area, name='Urban or Rural Area',

                     marker_colors= cafe_colors, textinfo='label+percent', insidetextorientation='radial'), 1, 2)





fig = go.Figure(fig)

fig.show()