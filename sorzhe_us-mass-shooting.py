# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

from geopy.geocoders import Nominatim

import geopy



matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/Mass Shootings Dataset.csv"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
d = pd.read_csv("../input/Mass Shootings Dataset.csv", encoding='latin-1')

d[['Latitude', 'Longitude']] = d[['Latitude', 'Longitude']].fillna(0)

#d.shape

d.head()


df = pd.DataFrame(d.Date)

df = pd.DataFrame(df.Date.str.split('/').tolist(), columns=['month', 'date', 'year']).join(d['Total victims'])

df = pd.DataFrame(df.groupby('year', as_index=True).sum())

#df

df.plot(kind = 'bar', figsize=(10,9), color='r')
d.info()
#Build the rating of total victims of shooting by state.



#Lets convert locations on States and clean up it

states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'D.C': 'District of Columbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'North Carolina',

        'ND': 'North Dakota',

        'NE': 'Nebraska',

        'NH': 'New Hampshire',

        'NJ': 'New Jersey',

        'NM': 'New Mexico',

        'NV': 'Nevada',

        'NY': 'New York',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'Rhode Island',

        'SC': 'South Carolina',

        'SD': 'South Dakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'Virgin Islands',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'West Virginia',

        'WY': 'Wyoming'

}



vd = pd.DataFrame(d.Location.str.split(',', expand=False).tolist()).replace({' ': None})

vd['State'] = (vd[4].fillna(vd[3]).fillna(vd[2]).fillna(vd[1]).fillna(vd[0])).str.strip()

vd = vd.join(d['Total victims'])



for state in vd.State.values:

    try:

        if state in states.keys():

            vd.State.replace({state: states[state]}, inplace=True)

        elif type(int(state)) == int:

            vd.State.replace({state: 'Kentucky'}, inplace=True)

        else:

            pass

    except: ValueError



vd = pd.DataFrame(vd.groupby('State', as_index=False).sum())

vd = vd.sort_values(by='Total victims', ascending=False)

#vd.info()

vd.plot(kind = 'bar', x='State', y='Total victims', figsize=(15,12), color='b')
#Set the coordinates instead NaN values



md = pd.DataFrame(d.Location.str.split(',', expand=False).tolist()).replace({' ': None})

md = md.join(d[['Latitude', 'Longitude', 'Total victims']])

zero_c = md[md.Latitude == 0]

coordinates = {'Las Vegas': (36.1662859, -115.149225), 'San Francisco': (37.7792808, -122.4192363), 'Tunkhannock': (41.5385159, -75.946844), 'Orlando': (28.5421232, -81.3790475), 'Kirkersville': (39.9595081, -82.5957182), 'Fresno': (36.7295295, -119.708861260756), 'Fort Lauderdale': (26.1223084, -80.1433786), 'Burlington': (44.4723989, -73.2114941), 'Baton Rouge': (30.4507462, -91.154551), 'Dallas': (32.7762719, -96.7968559), 'Hesston': (38.1383437, -97.4314267), 'Kalamazoo County': (42.2494968, -85.5372797), 'San Bernardino': (34.1083449, -117.2897652), 'Colorado Springs': (38.8339578, -104.8253485), 'Roseburg': (53.5314978, 10.6265906), 'Menasha': (44.2022293, -88.4465361), 'Santa Barbara': (34.4221319, -119.7026673), 'Fort Hood': (31.20143875, -97.7717578135356)}

for i in zero_c.index:

    city = zero_c[0][i]

    md.set_value(i, 'Latitude', coordinates[city][0])

    md.set_value(i, 'Longitude', coordinates[city][1])

#md.info()

#Visualize mass shootings on US map



from mpl_toolkits.basemap import Basemap as Basemap

from matplotlib.colors import rgb2hex

from matplotlib.patches import Polygon

from matplotlib.cm import ScalarMappable



fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

us_map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)



us_map.drawcoastlines() #zorder=3

us_map.drawmapboundary(zorder=0) #fill_color='#9fdbff'

us_map.fillcontinents(color='#ffffff',zorder=1) #,lake_color='#9fdbff',alpha=1

us_map.drawcountries(linewidth=1.5, color='darkblue') #color='darkblue'

us_map.drawstates(zorder=3) #zorder=3

#plt.show()



#Set county location values, shooting level values, marker sizes (according to county size), colormap and title 

x, y = us_map(md.Longitude.tolist(), md.Latitude.tolist())

colors = (md['Total victims']).tolist()

sizes = (md['Total victims']*2).tolist()

cmap = plt.cm.YlOrRd

sm = ScalarMappable(cmap=cmap)

plt.title('US shooting victims')



scatter = ax.scatter(x,y,s=sizes,c='r',cmap=cmap,alpha=1,edgecolors='face',marker='o',zorder=3)

plt.show()
d.fillna(value='Unknown', inplace=True)
import seaborn as sns

test_table = pd.DataFrame(d[['Gender', 'Race']], columns=['Gender', 'Race'])

test_table.Gender.replace({'M': 'Male', 'M/F': 'Male/Female'}, inplace=True)

test_table.Race.replace({'white': 'White', 'black': 'Black'}, inplace=True)

gender_table = pd.crosstab(test_table['Race'], test_table['Gender'])

gender_table

corr = gender_table.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

sns.plt.suptitle('Correlation between gender')
race_table =pd.crosstab(test_table['Gender'], test_table['Race'])

corr = race_table.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


d['Date'] = pd.to_datetime(d['Date'])



days = pd.DataFrame({'days': d.Date.dt.dayofweek, 'fatalities': d['Fatalities']})

days = days.groupby(by='days').sum()

days.plot(kind = 'bar',  figsize=(15,12), color='#456543')

plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=14, rotation=0)

plt.yticks(fontsize=14)

plt.xlabel('Days of week', fontsize=18)

plt.ylabel('Total fatalicies', fontsize=18)

plt.title('Deadly days', fontsize=20)
weeks = pd.DataFrame({'weeks': d.Date.dt.weekofyear, 'fatalities': d['Fatalities']})

weeks = weeks.groupby(by='weeks').sum()

weeks.plot(kind = 'bar',  figsize=(15,12), color='#ff6243')

plt.xticks( fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.xlabel('Week of year', fontsize=18)

plt.ylabel('Total fatalicies', fontsize=18)

plt.title('Deadly weeks', fontsize=20)
import calendar

months = pd.DataFrame({'months': d.Date.dt.month, 'fatalities': d['Fatalities']})

months = months.groupby(by='months').sum()

months.plot(kind = 'bar',  figsize=(15,12), color='#1abf4d')

plt.xticks(np.arange(12), calendar.month_name[1:13], fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.xlabel('Month of year', fontsize=18)

plt.ylabel('Total fatalicies', fontsize=18)

plt.title('Deadly months', fontsize=20)