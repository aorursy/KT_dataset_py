import datetime, warnings, scipy 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib.patches import ConnectionPatch

from collections import OrderedDict

from matplotlib.gridspec import GridSpec

from mpl_toolkits.basemap import Basemap

from sklearn import metrics, linear_model

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from scipy.optimize import curve_fit

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr"

pd.options.display.max_columns = 50

%matplotlib inline

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/flights.csv', low_memory=False)

airports = pd.read_csv("../input/airports.csv")

airlines = pd.read_csv("../input/airlines.csv")

print('Dataframe dimensions:', df.shape)
# Info about types and null values

tab_info=pd.DataFrame(df.dtypes).T.rename(index={0: 'COLUMN TYPES'})

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'NULL VALUES (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

tab_info

count_flights = df['ORIGIN_AIRPORT'].value_counts()

plt.figure(figsize=(11,11))



# define properties of markers and labels

colors = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']

size_limits = [1, 100, 1000, 10000, 100000, 1000000]

labels = []

for i in range(len(size_limits)-1):

    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 

#____________________________________________________________

'''

map = Basemap(resolution='i',llcrnrlon=-180, urcrnrlon=-50,

              llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0,)

map.shadedrelief()

map.drawcoastlines()

map.drawcountries(linewidth = 3)

map.drawstates(color='0.3')

'''

# use low resolution coastlines.

map = Basemap(resolution='i',llcrnrlon=-180, urcrnrlon=-50,

              llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0,)

# draw coastlines, country boundaries, fill continents.

map.drawmapboundary(fill_color='#99ffff')

map.drawcoastlines(linewidth=0.25)

map.drawcountries(linewidth=0.25)

map.fillcontinents(color='#cc9966',lake_color='aqua')

#_____________________

# put airports on map

for index, (code, y,x) in airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']].iterrows():

    x, y = map(x, y)

    isize = [i for i, val in enumerate(size_limits) if val < count_flights[code]]

    ind = isize[-1]

    map.plot(x, y, marker='o', markersize = ind+5, markeredgewidth = 1, color = colors[ind],

             markeredgecolor='k', label = labels[ind])

#_____________________________________________

# remove duplicate labels and set their order

handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

key_order = ('1 <.< 100', '100 <.< 1000', '1000 <.< 10000',

             '10000 <.< 100000', '100000 <.< 1000000')

new_label = OrderedDict()

for key in key_order:

    new_label[key] = by_label[key]

plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},

           title='Number of flights per year', frameon = True, framealpha = 1)

plt.show()
# Test only December 2015

df = df[df['MONTH'] == 12]
df['DATE'] = pd.to_datetime(df[['YEAR','MONTH', 'DAY']])
#_________________________________________________________

# Function that convert the 'HHMM' string to datetime.time

def format_heure(chaine):

    if pd.isnull(chaine):

        return np.nan

    else:

        if chaine == 2400: chaine = 0

        chaine = "{0:04d}".format(int(chaine))

        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))

        return heure

#_____________________________________________________________________

# Function that combines a date and time to produce a datetime.datetime

def combine_date_heure(x):

    if pd.isnull(x[0]) or pd.isnull(x[1]):

        return np.nan

    else:

        return datetime.datetime.combine(x[0],x[1])

#_______________________________________________________________________________

# Function that combine two columns of the dataframe to create a datetime format

def create_flight_time(df, col):    

    liste = []

    for index, cols in df[['DATE', col]].iterrows():    

        if pd.isnull(cols[1]):

            liste.append(np.nan)

        elif float(cols[1]) == 2400:

            cols[0] += datetime.timedelta(days=1)

            cols[1] = datetime.time(0,0)

            liste.append(combine_date_heure(cols))

        else:

            cols[1] = format_heure(cols[1])

            liste.append(combine_date_heure(cols))

    return pd.Series(liste)
df['SCHEDULED_DEPARTURE'] = create_flight_time(df, 'SCHEDULED_DEPARTURE')

df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_heure)

df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_heure)

df['ARRIVAL_TIME'] = df['ARRIVAL_TIME'].apply(format_heure)

#__________________________________________________________________________

df.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME',

             'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]
variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR', 

                       'MONTH','DAY','DAY_OF_WEEK','DATE', 'AIR_SYSTEM_DELAY',

                       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',

                       'WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',

                       'FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME']

df.drop(variables_to_remove, axis = 1, inplace = True)

df = df[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',

        'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',

        'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',

        'SCHEDULED_TIME', 'ELAPSED_TIME']]

df.head(5)