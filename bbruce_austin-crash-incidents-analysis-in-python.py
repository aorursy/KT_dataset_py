# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from bokeh.plotting import figure, show, output_notebook

from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper, CategoricalColorMapper

from bokeh.tile_providers import CARTODBPOSITRON_RETINA, STAMEN_TONER

from bokeh.palettes import Greens9

from bokeh.models.widgets import Panel, Tabs

from bokeh.layouts import column



output_notebook()
# read data

file_path = ['../input/austin_incidents_2016.txt', 

             '../input/austin_incidents_2011.txt',

             '../input/austin_incidents_2010.txt', 

             '../input/austin_incidents_2009.txt',

             '../input/austin_incidents_2008.txt']



austin = pd.concat((pd.read_csv(path, delimiter=',') for path in file_path))



# drop uninterested columns and NaN in ['descript' and 'location] columns

austin = (austin.dropna(subset=['descript', 'location'], axis=0)

          .drop(['c_time', 'c_time1', 'c_time2', 'c_time3', 'time_char','time'], axis=1))



# drop data in 2015 becaue we shouldn't have data from 2015

austin = austin[austin.date.str[:4] != '2015']
# get top 10 incidents

top10_list = austin.descript.value_counts().head(10).index.tolist() 

top10 = austin.copy()[austin.descript.str.contains('|'.join(top10_list))]



# all 'DWI' incidents are considered as the same `DWI`

top10.descript = top10.descript.apply(lambda row: 'DWI' if 'DWI' in row else row)

top10.descript = top10.descript.apply(lambda row: 'THEFT' if 'THEFT' in row else row)



# count top 10 incidents

top10['yr'] = top10.date.str[:4]

top10 = top10.groupby(['yr', 'descript'], as_index=False)['unique_key'].count()



# add x column for plotting

top10['x'] = [i for i in range(10)] * 5



# change 2016 to 2012 for plotting

top10.yr = top10.yr.apply(lambda row: '2012' if row == '2016' else row)



# calculate percentage of incidents in each yr

temp = top10.groupby(['yr'], as_index=False)['unique_key'].sum().rename(columns={'unique_key': 'p'})

top10 = top10.merge(temp, left_on='yr', right_on='yr', how='left')

top10.p = (top10.unique_key / top10.p * 100 )#.round(2)
# create `ColumnDataSource`

cds = ColumnDataSource(top10)



# set hover tool

hover = HoverTool(tooltips=[('Number of Records', '@unique_key'), 

                            ('Percentage (%)', '@p')])



# initiate a graph

p = figure(title='Top 10 Incidents in Austin (2008-2011, 2016)', 

           plot_width=600, plot_height=400,

           x_range=[2007.6, 2012.4], y_range=[-.6, 9.6],

           x_axis_location='above', outline_line_color=None,

           tools=['crosshair', hover], toolbar_location=None)



# set color mapper

mapper = LogColorMapper(palette=Greens9[::-1])



# add data point

p.circle('yr', 'x', source=cds, size=29, 

         line_width=3, line_color=None, 

         fill_color={'field': 'unique_key', 'transform': mapper},

         hover_line_color='black',

         hover_fill_color={'field': 'unique_key', 'transform': mapper},)



# graph properties

p.yaxis.ticker = [i for i in range(10)]

p.yaxis.major_label_overrides = {y: d for y, d in enumerate(top10.descript.unique())}

p.xaxis.major_label_overrides = {'2012': '2016'}

p.yaxis.axis_label = 'Description'

p.xaxis.axis_label = 'Year'

p.xgrid.grid_line_color = None

p.axis.minor_tick_line_color = None

p.axis.axis_line_color = None



show(p)
# get 'CRASH/LEAVING THE SCENE' incident

crash = austin.copy()[austin.descript == 'CRASH/LEAVING THE SCENE']



# count incidents

df = []

years = ['2008', '2009', '2010', '2011', '2016']

for i in years:

    temp = crash.copy()[crash.date.str[:4] == i]

    temp['month'] = temp.date.str[5:7]

    temp = temp.groupby(['month'])['unique_key'].count()

    df.append(temp)



df = pd.concat(df, axis=1).fillna(0)

df.columns = ['2008', '2009', '2010', '2011', '2016']

df['2016'] = df['2016'].astype(int)
# deal with `datetime` object

temp = df.reset_index().rename(columns={'index': 'month'})

temp.month = pd.to_datetime(temp.month, format='%m').dt.strftime('%b')



# barchart

colors = ['#203b77', '#274891', '#2d53a8', '#3661c1', '#eff4ff']

ax = temp.plot.bar(x='month', stacked=True, figsize=(12,5),

                   color=colors, fontsize=12)



# add annotations

hgts = []

for idx, rect in enumerate(ax.patches):

    

    if idx > 55: break

    

    # height and width of each bar

    hgt, wgt = rect.get_height(), rect.get_x() + rect.get_width()/2

    

    # accumulate height

    if idx > 11: 

        hgt += hgts[idx-12]

    hgts.append(hgt)

    

    ax.annotate(hgts[idx], (wgt, hgt), size=9, color='w', alpha=.5,

                ha='center', va='center', 

                xytext=(0, -15), textcoords='offset points')



# set graph properties

_ = [spine.set_visible(False) for spine in plt.gca().spines.values()]

plt.tick_params(bottom='off', left='off', labelleft='on', labelbottom='on')

plt.title('Monthly CRASH/LEAVING THE SCENE Incident in Austin', fontsize=15)

plt.xlabel('Month', fontsize=12)

plt.ylabel('Number of Records', fontsize=12)

plt.xticks(rotation=20)

plt.legend(frameon=False)

plt.margins(.01)



plt.show()
temp = pd.melt(df)

temp['date'] = pd.to_datetime(temp.variable + '-' + [str(i+1) for i in range(12)]*5)

temp = temp[['value', 'date']].set_index(['date'])

temp = temp[(temp!=0).any(axis=1)]



# subset 2008-2011, 2016

temp4yrs, temp2016 = temp[temp.index < '2016'], temp[temp.index > '2015']



# lineplot

ax = temp4yrs.plot(figsize=(12,5), color='#b23221', marker='o')

temp2016.plot(color='#7c1a0d', marker='o', ax=ax)



# set graph properties

_ = [spine.set_visible(False) for spine in plt.gca().spines.values()]

plt.tick_params(bottom='off', left='off', labelbottom='on')

plt.title('Yearly CRASH/LEAVING THE SCENE Incident in Austin', fontsize=15)

plt.xlabel('Month', fontsize=12)

plt.ylabel('Number of Records', fontsize=12)

plt.legend(['2008-2011', '2016'], frameon=False)

plt.xticks(rotation=20)



plt.show()
# get 'CRASH/LEAVING THE SCENE' incident

crash = austin.copy()[austin.descript == 'CRASH/LEAVING THE SCENE']



# deal with `datetime` object

crash.timestamp = pd.to_datetime(crash.timestamp)



# extract interested columns

geo = crash.copy()[['address', 'latitude', 'longitude', 'timestamp']]

geo['yr'] = geo.timestamp.dt.year



# map hour into 6 periods

# hr_mapper = {0: '22:00-01:59 (Midnight)', 

#              1: '02:00-05:59 (Dawn)', 

#              2: '06:00-09:59 (Morning)',

#              3: '10:00-13:59 (Midday)', 

#              4: '14:00-17:59 (Afternoon)', 

#              5: '18:00-21:59 (Evening)'}

hr_mapper = {0: '22:00-01:59', 1: '02:00-05:59', 2: '06:00-09:59',

             3: '10:00-13:59', 4: '14:00-17:59', 5: '18:00-21:59'}



geo['daynight'] = (geo.timestamp.dt.hour-22)/4

geo['daynight'] = [int(p+6) if p<0. else int(p) for p in geo.daynight]

geo['daynight'] = geo['daynight'].map(hr_mapper)



# count

geo = (geo.merge(geo.groupby(['yr'])['daynight'].value_counts().to_frame(),

                 how='left', left_on=['yr', 'daynight'], right_index=True)

       .rename(columns={'daynight_x': 'daynight', 'daynight_y': 'counts'}))



# Convert coordinates

# define functions for coordinate projection

import math



def lgn2x(a):

    return a * (math.pi/180) * 6378137



def lat2y(a):

    return math.log(math.tan(a * (math.pi/180)/2 + math.pi/4)) * 6378137



geo['x'] = geo.longitude.apply(lambda row: lgn2x(row))

geo['y'] = geo.latitude.apply(lambda row: lat2y(row))



# extract interested columns

geo = geo[['address', 'x', 'y', 'timestamp', 'daynight', 'counts', 'yr']]
factors = sorted(geo.daynight.unique())

mapper = CategoricalColorMapper(factors=factors,

                                palette=['navy', 'red', 'orange', 'brown', 'blue', 'darkviolet'])

colors = {'field': 'daynight', 'transform': mapper}



graphs = []

for i in [2008, 2009, 2010, 2011, 2016]:

    

    # create `ColumnDataSource`

    temp = geo.copy()[geo.yr==i]

    temp.timestamp = temp.timestamp.astype('str')

    cds = ColumnDataSource(temp)

    

    # customize hover tool

    hover = HoverTool(tooltips= [('Timestamp', '@timestamp'), ('Address', '@address')])

    

    # initiate a figure, add dots, overlay map

    p = figure(title='Geo Map of CRASH/LEAVING THE SCENE incident in Austin (2008-2011, 2016 Aug.)',

               plot_width=650, plot_height=520, 

               x_range=(-1.093e7, -1.084e7), y_range=(3.510e6, 3.580e6),

               x_axis_location=None, y_axis_location=None,

               tools=[hover, 'pan', 'wheel_zoom', 'crosshair', 'reset'], 

               toolbar_location='left')

    p.circle('x', 'y', source=cds, size=7, color=colors, fill_alpha=.5, legend='daynight')

    p.add_tile(STAMEN_TONER)

    

    # intiate a figure, add bars

    bar = figure(title='Click bar below to view the specific time period',

                 plot_width=650, plot_height=100, 

                 y_range=[0, 650], x_range=factors, 

                 outline_line_color=None,

                 tools='tap', toolbar_location=None)

    bar.vbar(x='daynight', top='counts', source=cds, width=.4, color=colors,

             selection_color='black', nonselection_fill_alpha=.1, nonselection_fill_color='gray')

    

    # graph properties

#     bar.xaxis.axis_label = 'Time Period (every 4hrs)'

    bar.yaxis.axis_label = 'Counts'

    bar.axis.minor_tick_line_color = None

    bar.xgrid.grid_line_color = None

    

    # build a tab

    tab = Panel(child=column(p, bar), title=str(i))

    graphs.append(tab)



layout = Tabs(tabs=graphs)



show(layout)