# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import json

# racing chart
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from random import randint
import colorsys
import matplotlib.colors as mc
import re

# Disable the warnings
import warnings
warnings.filterwarnings("ignore")


%matplotlib inline
# Import data
world_data = pd.read_csv('../input/covid19-week4/train.csv')
by_counties = pd.read_csv('../input/nytimes/us-counties.csv', index_col = 0)
by_states = pd.read_csv('../input/nytimes/us-states.csv')
world_data.head(3)
# Rename the column 'Date'
world_data = world_data.rename(columns={'Date': 'Date_worldwide'})
# The graph will only show countries with confirmed cases
# Each animated graph will be grouped by the date and the country
df_countrydate = world_data[world_data['ConfirmedCases']>0]
df_countrydate = df_countrydate.groupby(['Date_worldwide','Country_Region']).sum().reset_index()
# Create the Choropleth

fig = px.choropleth(df_countrydate, 
                    locations = 'Country_Region', # Spatial coordinates
                    locationmode = 'country names',
                    color = 'ConfirmedCases', 
                    color_continuous_scale = 'tempo',
                    hover_name = 'Country_Region',
                    animation_frame = 'Date_worldwide',
                   )

fig.update_layout(
    title_text = 'Reported Confirmed Cases Worldwide between 1/22/2010 and 04/10/2020',
    title_x = 0.5,
    geo = dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
# Create the Choropleth
fig = px.choropleth(df_countrydate, 
                    locations = 'Country_Region', 
                    locationmode = 'country names', # Spatial coordinates
                    color = 'Fatalities', 
                    color_continuous_scale = 'reds',
                    hover_name = 'Country_Region', 
                    animation_frame = 'Date_worldwide',
                   )

fig.update_layout(
    title_text = 'Reported Fatalities Worldwide between 1/22/2010 and 04/10/2020',
    title_x = 0.5,
    geo = dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()

# change date format from string to datetime
by_states['date'] =  pd.to_datetime(by_states['date'], format='%Y-%m-%d')
# The racing bar chart is built by a series of charts grouped by 'state' along the date
by_states = by_states.groupby(['date', 'state']).sum().reset_index()
by_states['date'] = by_states['date'].astype('str') # The racing chart will go with string

# define a new dataframe for the racing chart
df = by_states[['date', 'state', 'cases', 'deaths']]
df.columns = ['date', 'state', 'value_confirmed', 'value_deaths']
fnames_list = df['date'].unique().tolist()

def random_color_generator(number_of_colors):
    random.seed(30)
    color = ["#"+''.join([random.choice('987654321ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

state_list = df['state'].unique().tolist()

num_of_elements = 10
def transform_color(color, amount = 0.5):

    try:
        c = mc.cnames[color]
    except:
        c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

random_hex_colors = []
for i in range(len(state_list)):
    random_hex_colors.append('#' + '%06X' % randint(0, 0xFFFFFF))

rgb_colors = [transform_color(i, 1) for i in random_hex_colors]
rgb_colors_opacity = [rgb_colors[x] + (0.825,) for x in range(len(rgb_colors))]
rgb_colors_dark = [transform_color(i, 1.12) for i in random_hex_colors]

normal_colors = dict(zip(state_list, rgb_colors_opacity))
dark_colors = dict(zip(state_list, rgb_colors_dark))
# Let's check the different available style sheets on Matplotlib
print(plt.style.available)
plt.style.use('grayscale')
fig, ax = plt.subplots(figsize = (36, 20))

def draw_barchart(current_date):
    dff = df[df['date'].eq(current_date)].sort_values(by='value_confirmed', ascending=True).tail(num_of_elements)
    ax.clear()
    
    ax.barh(dff['state'], dff['value_confirmed'], color=[normal_colors[p] for p in dff['state']],
                edgecolor =([dark_colors[x] for x in dff['state']]), linewidth = '6')
    dx = dff['value_confirmed'].max() / 200


    for i, (value, name) in enumerate(zip(dff['value_confirmed'], dff['state'])):
        ax.text(value + dx, 
                i + (num_of_elements / 50), '    ' + name,
                size = 32,
                ha = 'left',
                va = 'center',
                fontdict = {'fontname': 'Trebuchet MS'})

        ax.text(value + dx,
                i - (num_of_elements / 50), 
                f'    {value:,.0f}', 
                size = 32, 
                ha = 'left', 
                va = 'center')   
    
    
    time_unit_displayed = re.sub(r'\^(.*)', r'', str(current_date))
    ax.text(1, 
            0.5, 
            time_unit_displayed,
            transform = ax.transAxes, 
            color = '#777777',
            size = 55,
            ha = 'right', 
            weight = 'bold', 
            fontdict = {'fontname': 'Trebuchet MS'})

    ax.text(-0.005, 
            1.05, 
            'cases', 
            transform = ax.transAxes, 
            size = 40, 
            color = '#666666')

    ax.text(0.07, 
            1.1, 
            'Confirmed Cases in the United States from 2020-01-21 to 2020-04-13', 
            transform = ax.transAxes,
            size = 50, 
            weight = 'bold', 
            color = 'royalblue',
            ha = 'left')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis = 'x', colors = '#666666', labelsize = 28)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.margins(0, 0.01)
    ax.grid(which = 'major', axis = 'x', linestyle = '-')
    

    plt.locator_params(axis = 'x', nbins = 4)
    plt.box(False)
    plt.subplots_adjust(left = 0.075, right = 0.75, top = 0.825, bottom = 0.05, wspace = 0.2, hspace = 0.2)
    plt.box(False)    

draw_barchart('2020-04-13')
fig, ax = plt.subplots(figsize = (36, 20))
animator = animation.FuncAnimation(fig, draw_barchart, frames=fnames_list)
racing_chart = HTML(animator.to_jshtml())

racing_chart
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize = (36, 20))

def draw_barchart(current_date):
    dff = df[df['date'].eq(current_date)].sort_values(by='value_deaths', ascending=True).tail(num_of_elements)
    ax.clear()
    
    ax.barh(dff['state'], dff['value_deaths'], color=[normal_colors[p] for p in dff['state']],
                edgecolor =([dark_colors[x] for x in dff['state']]), linewidth = '6')
    dx = dff['value_deaths'].max() / 200


    for i, (value, name) in enumerate(zip(dff['value_deaths'], dff['state'])):
        ax.text(value + dx, 
                i + (num_of_elements / 50), '    ' + name,
                size = 32,
                ha = 'left',
                va = 'center',
                fontdict = {'fontname': 'Trebuchet MS'})

        ax.text(value + dx,
                i - (num_of_elements / 50), 
                f'    {value:,.0f}', 
                size = 32, 
                ha = 'left', 
                va = 'center')   
    
    
    time_unit_displayed = re.sub(r'\^(.*)', r'', str(current_date))
    ax.text(1, 
            0.5, 
            time_unit_displayed,
            transform = ax.transAxes, 
            color = '#777777',
            size = 55,
            ha = 'right', 
            weight = 'bold', 
            fontdict = {'fontname': 'Trebuchet MS'})

    ax.text(-0.005, 
            1.05, 
            'cases', 
            transform = ax.transAxes, 
            size = 40, 
            color = '#666666')

    ax.text(0.2, 
            1.1, 
            'Fatalities in US from 2020-01-21 to 2020-04-13', 
            transform = ax.transAxes,
            size = 50, 
            weight = 'bold',
            color = 'firebrick',
            ha = 'left')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis = 'x', colors = '#666666', labelsize = 28)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.margins(0, 0.01)
    ax.grid(which = 'major', axis = 'x', linestyle = '-')
    

    plt.locator_params(axis = 'x', nbins = 4)
    plt.box(False)
    plt.subplots_adjust(left = 0.075, right = 0.75, top = 0.825, bottom = 0.05, wspace = 0.2, hspace = 0.2)
    plt.box(False)    
draw_barchart('2020-04-13')
fig, ax = plt.subplots(figsize = (36, 20))
animator = animation.FuncAnimation(fig, draw_barchart, frames=fnames_list)
chart_race = HTML(animator.to_jshtml())
chart_race
def pie(ax, values, **kwargs):
    total = sum(values)
    def formatter(pct):
        return '{:0.0f}%'.format(pct*total/100)
        #return '${:0.0f}M\n({:0.1f}%)'.format(pct*total/100, pct)
    wedges, _, labels = ax.pie(values, autopct=formatter, **kwargs)
    return wedges

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle('Fatalities by Race/ Ethnicity from Covid-19 in New York City', color='black', fontsize=22)

ax.axis('equal')

width = 0.35
kwargs = dict(colors=['#9999ff', '#ffff99', '#99ff99', '#ff9999', '#99ccff'], startangle=90)

outside = pie(ax, [29, 22, 32, 14, 3], radius=1, pctdistance=1-width/2, **kwargs)
inside = pie(ax, [34, 28, 27, 7, 4], radius=1-width,
             pctdistance=1 - (width/2) / (1-width), **kwargs)
plt.setp(inside + outside, width=width, edgecolor='white')

ax.legend(inside[:], ['Hispanic', 'Black', 'White', 'Asian','Other' ], prop={'size': 13}, frameon=False)

kwargs = dict(size=15, va='center', fontweight='bold')
ax.text(0, 0, '% of Fatalities', ha='center', color='red',
        bbox=dict(boxstyle='round', facecolor='silver', edgecolor='none'),
        **kwargs)
ax.annotate('% of NYC Population', (0, 0), color='blue', xytext=(np.radians(-45), 1.1) ,
            bbox=dict(boxstyle='round', facecolor='silver', edgecolor='none'),
            textcoords='polar', ha='left', **kwargs)

plt.show()
# the bar chart will use the time as index and the x-axis
by_counties.index = pd.to_datetime(by_counties.index)

# take of the unnecessary columns
by_counties = by_counties.drop(columns=['fips', 'state'])
by_counties = by_counties.loc[by_counties['county'] == 'New York City']
plt.rcParams['figure.figsize']=(20,10) # set the figure size # set the figure size
plt.style.use('fivethirtyeight') # using the fivethirtyeight matplotlib theme

# Divide the timeframe into three segments for different colors
first = by_counties[(by_counties.index >= '2020-03-01') & (by_counties.index < '2020-03-15')]
second = by_counties[(by_counties.index >= '2020-03-15') & (by_counties.index < '2020-03-31')]
third = by_counties[(by_counties.index >= '2020-04-01') & (by_counties.index < '2020-04-13')]

# Build our plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis
ax1.plot(by_counties.deaths) #plot the Revenue on axis #1

# the next few lines plot the fiscal year data as bar plots and changes the color for each.
ax2.bar(first.index, first.cases,width=0.5, alpha=0.5, color='green')
ax2.bar(second.index, second.cases,width=0.5, alpha=0.5, color='blue')
ax2.bar(third.index, third.cases,width=0.5, alpha=0.5, color='orange')

ax2.grid(b=False) # turn off grid #2

ax1.set_title('Confirmed Cases and Fatalites in NYC from 2020-03-01 to 2020-04-13)', color='black')
ax1.set_ylabel('Fatalies')
ax2.set_ylabel('Confirmed Cases')

# Set the x-axis labels to be more meaningful than just some random dates.
labels = ['2020-03-01','2020-03-15','2020-04-01'] 
ax1.set_xticks(labels)
ax1.axes.set_xticklabels(labels)