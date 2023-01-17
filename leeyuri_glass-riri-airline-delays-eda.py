import pandas as pd

import numpy as np

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

%matplotlib inline

import datetime, warnings, scipy 

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib.patches import ConnectionPatch

from collections import OrderedDict

from matplotlib.gridspec import GridSpec

from mpl_toolkits.basemap import Basemap

from scipy.optimize import curve_fit

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr"

pd.options.display.max_columns = 50

warnings.filterwarnings("ignore")
airlines = pd.read_csv('../input/airlines.csv')

airports= pd.read_csv('../input/airports.csv')

flights = pd.read_csv('../input/flights.csv', low_memory=False)

print (airlines.shape) # 14행 2열

print (airports.shape) # 332행 7열

print (flights.shape)  #5819079행 31열
#결항(canclelled)컬럼의 평균과 비행기의 개수의 곱

print (flights.shape[0]*flights.CANCELLED.mean())

print (flights.shape[0] - flights.CANCELLATION_REASON.isnull().sum().sum())
print (flights['ARRIVAL_DELAY'][flights['ARRIVAL_DELAY'] >= 15].count())

print (flights.shape[0] - flights.AIR_SYSTEM_DELAY.isnull().sum().sum())
flights_v1 = pd.merge(flights, airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')

flights_v1.drop('IATA_CODE', axis=1, inplace=True)

flights_v1.rename(columns={'AIRLINE_x': 'AIRLINE_CODE','AIRLINE_y': 'AIRLINE'}, inplace=True)
airport_mean_delays = pd.DataFrame(pd.Series(flights['ORIGIN_AIRPORT'].unique()))

airport_mean_delays.set_index(0, drop = True, inplace = True)

abbr_companies = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()

identify_airport = airports.set_index('IATA_CODE')['CITY'].to_dict()



# function that extract statistical parameters from a grouby objet:

def get_stats(group):

    return {'min': group.min(), 'max': group.max(),

            'count': group.count(), 'mean': group.mean()}

#___________________________________________________________



for carrier in abbr_companies.keys():

    fg1 = flights[flights['AIRLINE'] == carrier]

    test = fg1['DEPARTURE_DELAY'].groupby(flights['ORIGIN_AIRPORT']).apply(get_stats).unstack()

    airport_mean_delays[carrier] = test.loc[:, 'mean'] 
sns.set(context="paper")

fig = plt.figure(1, figsize=(12,15))



ax = fig.add_subplot(1,2,1)

subset = airport_mean_delays.iloc[:50,:].rename(columns = abbr_companies)

subset = subset.rename(index = identify_airport)

mask = subset.isnull()

sns.heatmap(subset, linewidths=0.05, cmap="YlGnBu", mask=mask, vmin = 0, vmax = 30)

plt.setp(ax.get_xticklabels(), fontsize=12, rotation = 88) ;

ax.yaxis.label.set_visible(False)



ax = fig.add_subplot(1,2,2)    

subset = airport_mean_delays.iloc[50:100,:].rename(columns = abbr_companies)

subset = subset.rename(index = identify_airport)

fig.text(0.5, 1.02, "Scale of Delays from origin airport", ha='center', fontsize = 20)

mask = subset.isnull()

sns.heatmap(subset, linewidths=0.05, cmap="YlGnBu", mask=mask, vmin = 0, vmax = 30)

plt.setp(ax.get_xticklabels(), fontsize=12, rotation = 88) ;

ax.yaxis.label.set_visible(False)



plt.tight_layout()
font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 15}

mpl.rc('font', **font)

import matplotlib.patches as mpatches

#__________________________________________________________________

# extract a subset of columns and redefine the airlines labeling 

fg2 = flights.loc[:, ['AIRLINE', 'DEPARTURE_DELAY']]

fg2['AIRLINE'] = fg2['AIRLINE'].replace(abbr_companies)

fig = plt.figure(1, figsize=(16,16))

gs=GridSpec(1,2)             

ax1=fig.add_subplot(gs[0,0]) 

ax2=fig.add_subplot(gs[0,1]) 



global_stats_d=flights['DEPARTURE_DELAY'].groupby(flights['AIRLINE']).apply(get_stats).unstack()

global_stats_d=global_stats_d.sort_values('count')



global_stats_a=flights['ARRIVAL_DELAY'].groupby(flights['AIRLINE']).apply(get_stats).unstack()

global_stats_a=global_stats_a.sort_values('count')



#----------------------------------------

# Pie chart nº1: mean delay at departure

#----------------------------------------

labels = [s for s in  global_stats_d.index]

sizes  = global_stats_d['mean'].values

sizes  = [max(s,0) for s in sizes]

explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]

patches, texts, autotexts = ax1.pie(sizes, explode = explode, labels = labels, shadow=False, startangle=0, 

                                    autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))

for i in range(len(abbr_companies)): 

    texts[i].set_fontsize(18)

ax1.axis('equal')

ax1.set_title('Mean delay at departure', bbox={'facecolor':'midnightblue', 'pad':10},

              color='w', fontsize=18)



#----------------------------------------

# Pie chart nº2: mean delay at arrival

#----------------------------------------

labels = [s for s in  global_stats_a.index]

sizes  = global_stats_a['mean'].values

sizes  = [max(s,0) for s in sizes]

explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]

patches, texts, autotexts = ax2.pie(sizes, explode = explode, labels = labels, shadow=False, startangle=0,

                                     autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))

for i in range(len(abbr_companies)): 

    texts[i].set_fontsize(18)

ax2.axis('equal')

ax2.set_title('Mean delay at arrival', bbox={'facecolor':'midnightblue', 'pad':10},

              color='w', fontsize=18)