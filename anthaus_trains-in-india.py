# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium

from folium.map import Icon



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_trains = pd.read_csv('../input/indiantrains/All_Indian_Trains.csv')

df_cities = pd.read_csv("../input/top-500-indian-cities/cities_r2.csv")

df_wcities = pd.read_csv("../input/world-cities-database/worldcitiespop.csv")
sub_wcities = pd.concat([df_wcities[df_wcities['Country'] == 'in'], df_wcities[df_wcities['Country'] == 'bd'] ,df_wcities[df_wcities['Country'] == 'bt'],df_wcities[df_wcities['Country'] == 'np'],df_wcities[df_wcities['Country'] == 'pk']])
df_trains.head(10)
len(df_trains)
sub_wcities.head(10)
len(sub_wcities[sub_wcities['Country'] == 'in'])
df_cities.head(10)
''' Here we will correct some mispellings that had been discovered during the analysis, but deleted from the notebook,

    for the sake of readibility. '''



def corr(name):

    if name == 'Velankanni' or name == 'Vellankanni':

        return 'Velanganni'

    elif name == 'Raxual Junction':

        return 'Raxaul Junction'

    elif name == 'Alipur Duar Junction':

        return 'Alipurduar Junction'

    elif name == 'Chamarajanagar':

        return 'Chamarajnagar'

    elif name == 'Dehradun':

        return 'Dehra Dun'

    elif name == 'Eranakulam Junction':

        return 'Ernakulam Junction'

    elif name == 'Machelipatnam':

        return 'Machilipatnam'

    elif name == 'Metupalaiyam':

        return 'Mettupalaiyam'

    elif name == 'Mathura Junction':

        return 'Vrindavan'               # This one is because the World Cities dataset provided Mathura in Andaman Islands first.

    elif name == 'Murkeongselek':

        return 'Murkong Selek'

    elif name == 'Nagarsol':

        return 'Nagarsul'

    elif name == 'New Delhi':

        return 'Newdelhi'

    elif name == 'Tiruchchirapali':

        return 'Tiruchchirappalli'

    elif name == 'Villuparam Junction':

        return 'Villupuram Junction'

    elif name == 'Vishakapatnam':

        return 'Vishakhapatnam'

    else:

        return name
ds = df_trains['Starts'].apply(corr)

de = df_trains['Ends'].apply(corr)



df_trains_aug = pd.DataFrame()      # Will be used to draw the map

df_trains_aug['Train no.'] = df_trains['Train no.']

df_trains_aug['Train name'] = df_trains['Train name']

df_trains_aug['Starts'] = ds

df_trains_aug['Ends'] = de



df_trains_aug.head(10)
df_stations = pd.DataFrame()       # Will group info about all the stations

sta_name = []

sta_city = []

sta_lat = []

sta_long = []

sta_starts = []

sta_ends = []

sta_trains = []

sta_state = []

sta_country = []

unfound = []

stations_set = set(df_trains_aug['Starts']).union(set(df_trains_aug['Ends']))
#sub_wcities[sub_wcities['City'] == 'adirampatnam']['Latitude'].to_numpy()[0]
for s in stations_set:

    found = False

    for w in sub_wcities['City']:

        if not found:

            if s.lower() in str(w).split(' ') or str(w) in s.lower().split(' ') or str(w) == s.lower():

                sta_name.append(s)

                sta_city.append(str(w))

                sta_lat.append(sub_wcities[sub_wcities['City'] == str(w)]['Latitude'].to_numpy()[0])

                sta_long.append(sub_wcities[sub_wcities['City'] == str(w)]['Longitude'].to_numpy()[0])

                sta_starts.append(len(df_trains_aug[df_trains_aug['Starts'] == s]))

                sta_ends.append(len(df_trains_aug[df_trains_aug['Ends'] == s]))

                sta_trains.append(len(df_trains_aug[df_trains_aug['Starts'] == s]) + len(df_trains_aug[df_trains_aug['Ends'] == s]))

                sta_state.append(sub_wcities[sub_wcities['City'] == str(w)]['Region'].to_numpy()[0])

                sta_country.append(sub_wcities[sub_wcities['City'] == str(w)]['Country'].to_numpy()[0])

                found = True

    if not found:

        unfound.append(s)
sta_starts[:10]

sta_ends[:10]
len(unfound)
unfound
manual_handle = {'Chirmiri':'korea', 'Manduadih':'varanasi', 'Sadulpur Junction':'churu', 'Manuguru':'kothagudem', 'Mayiladuturai J':'mayuram', 'Sengottai':'tenkasi',

                'Kochuveli':'thiruvananthapuram', 'Patliputra':'danapur', 'Chamarajnagar':'mysore', 'C Shahumharaj T':'kolhapur', 'Lokmanyatilak T':'kurla', 'Gevra Road':'korba',

                'Singrauli':'churki', 'Shmata V D Ktra':'dudura', 'New Alipurdaur':'alipur duar', 'Alipurduar Junction':'alipur duar', 'Habibganj':'bhopal', 'Banaswadi':'bangalore', 'Jhajha':'jamui',

                'Sawantwadi Road':'talavada', 'H Nizamuddin':'delhi', 'Naharlagun':'itanagar', 'Nilaje':'mumbai', 'Khairthal':'alwar', 'Udhna Junction':'surat', 'Kirandul':'dantewara',

                'Kacheguda':'hyderabad', 'Belampalli':'mancherial', 'Radhikapur':'raiganj', 'Borivali':'mumbai', 'Dekargaon':'tezpur', 'Newdelhi': 'new delhi'}



for s in manual_handle.keys():

    sta_name.append(s)

    sta_city.append(manual_handle[s])

    sta_lat.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Latitude'].to_numpy()[0])

    sta_long.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Longitude'].to_numpy()[0])

    sta_starts.append(len(df_trains_aug[df_trains_aug['Starts'] == s]))

    sta_ends.append(len(df_trains_aug[df_trains_aug['Ends'] == s]))

    sta_trains.append(len(df_trains_aug[df_trains_aug['Starts'] == s]) + len(df_trains_aug[df_trains_aug['Ends'] == s]))

    sta_state.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Region'].to_numpy()[0])

    sta_country.append(sub_wcities[sub_wcities['City'] == manual_handle[s]]['Country'].to_numpy()[0])
df_stations['name'] = sta_name

df_stations['city'] = sta_city

df_stations['latitude'] = sta_lat

df_stations['longitude'] = sta_long

df_stations['nb_starts'] = sta_starts

df_stations['nb_ends'] = sta_ends

df_stations['nb_trains'] = sta_trains

df_stations['state'] = sta_state

df_stations['country'] = sta_country
df_stations.describe()
stations_map = folium.Map(location=[22.05, 78.94], zoom_start=4.5)

for idx, row in df_stations.iterrows():

    c = 'mediumpurple'

    if row['nb_ends'] == 0:

        c = 'royalblue'

    if row['nb_starts'] == 0:

        c = 'deeppink'

    folium.Circle(location=[row['latitude'], row['longitude']], radius=1 + 400 * row['nb_trains'], color = c, fill = True, popup = row['name']).add_to(stations_map)

stations_map
df_stations.sort_values('nb_trains',ascending=False).head(10)
howrah_lines = folium.Map(location=[22.59,88.31], zoom_start=4.5)

x0 = df_stations[df_stations['name'] == 'Howrah Junction']['latitude'].to_numpy()[0]

x1 = df_stations[df_stations['name'] == 'Howrah Junction']['longitude'].to_numpy()[0]

folium.Marker(location=(x0, x1), icon=Icon(color='purple', icon='train')).add_to(howrah_lines)

for idx, row in df_trains_aug.iterrows():

    if row['Starts'] == 'Howrah Junction':

        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]

        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]

        folium.Marker(location=(y0, y1), icon=Icon(color='green', icon='train')).add_to(howrah_lines)

    elif row['Ends'] == 'Howrah Junction':

        y0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]

        y1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]

        folium.Marker(location=(y0, y1), icon=Icon(color='orange', icon='train')).add_to(howrah_lines)

howrah_lines
foreign_lines = folium.Map(location=[22.05, 78.94], zoom_start=4.5)

for idx, row in df_trains_aug.iterrows():

    if df_stations[df_stations['name'] == row['Starts']]['country'].to_numpy()[0] != 'in':

        x0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]

        x1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]

        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]

        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]

        folium.PolyLine(locations=[(x0, x1),(y0, y1)], color='limegreen').add_to(foreign_lines)

    elif df_stations[df_stations['name'] == row['Ends']]['country'].to_numpy()[0] != 'in':

        x0 = df_stations[df_stations['name'] == [row['Starts']][0]]['latitude'].to_numpy()[0]

        x1 = df_stations[df_stations['name'] == [row['Starts']][0]]['longitude'].to_numpy()[0]

        y0 = df_stations[df_stations['name'] == [row['Ends']][0]]['latitude'].to_numpy()[0]

        y1 = df_stations[df_stations['name'] == [row['Ends']][0]]['longitude'].to_numpy()[0]

        folium.PolyLine(locations=[(x0, x1),(y0, y1)], color='darkorange').add_to(foreign_lines)

foreign_lines
df_cities_stations = pd.DataFrame()

cs_name = []

cs_nb_stations = []

cs_nb_start_trains = []

cs_nb_end_trains = []

cs_nb_trains = []

cs_population = []

cs_literacy = []

cs_literacy_gap = []

cs_graduate = []

cs_state = []

cs_latitude = []

cs_longitude = []
stat_cities = set(df_stations['city'])

len(stat_cities)
for sc in stat_cities:

    for C in df_cities['name_of_city']:

        if sc in C.lower().split(' ') or C.lower() in sc.split(' '):

            subset = df_stations[df_stations['city'] == sc]

            cs_name.append(C)

            cs_nb_stations.append(len(subset))

            cs_nb_start_trains.append(sum(subset['nb_starts']))

            cs_nb_end_trains.append(sum(subset['nb_ends']))

            cs_nb_trains.append(sum(subset['nb_trains']))

            cs_population.append(df_cities[df_cities['name_of_city'] == C]['population_total'].to_numpy()[0])

            cs_literacy.append(df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_total'].to_numpy()[0])

            cs_literacy_gap.append(df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_male'].to_numpy()[0] - df_cities[df_cities['name_of_city'] == C]['effective_literacy_rate_female'].to_numpy()[0])

            cs_graduate.append(df_cities[df_cities['name_of_city'] == C]['total_graduates'].to_numpy()[0])

            cs_state.append(df_cities[df_cities['name_of_city'] == C]['state_name'].to_numpy()[0])

            cs_latitude.append(df_cities[df_cities['name_of_city'] == C]['location'].to_numpy()[0].split(',')[0])

            cs_longitude.append(df_cities[df_cities['name_of_city'] == C]['location'].to_numpy()[0].split(',')[1])



df_cities_stations['name'] = cs_name

df_cities_stations['nb_stations'] = cs_nb_stations

df_cities_stations['nb_start_trains'] = cs_nb_start_trains

df_cities_stations['nb_end_trains'] = cs_nb_end_trains

df_cities_stations['nb_trains'] = cs_nb_trains

df_cities_stations['population'] = cs_population

df_cities_stations['literacy'] = cs_literacy

df_cities_stations['literacy_gap'] = cs_literacy_gap

df_cities_stations['graduate'] = cs_graduate

df_cities_stations['state'] = cs_state

df_cities_stations['latitude'] = cs_latitude

df_cities_stations['longitude'] = cs_longitude
df_cities_stations.head(10)
df_cities_stations.sort_values('nb_stations', ascending=False).head(20)
df_cities_stations = df_cities_stations.drop([82,83,148,149])

df_cities_stations.sort_values('nb_stations', ascending=False)
df_stations[df_stations['city'] == 'new delhi']
for idx, row in df_cities.iterrows():

    if 'new delhi' in row['name_of_city'].lower():

        df_cities_stations = df_cities_stations.append(

                                {'name':row['name_of_city'],

                                'nb_stations':1,

                                'nb_start_trains':120,

                                'nb_end_trains':123,

                                'nb_trains':243,

                                'population':df_cities[df_cities['name_of_city'] == row['name_of_city']]['population_total'].to_numpy()[0],

                                'literacy':df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_total'].to_numpy()[0],

                                'literacy_gap':df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_male'].to_numpy()[0] - df_cities[df_cities['name_of_city'] == row['name_of_city']]['effective_literacy_rate_female'].to_numpy()[0],

                                'graduate':df_cities[df_cities['name_of_city'] == row['name_of_city']]['total_graduates'].to_numpy()[0],

                                'state':df_cities[df_cities['name_of_city'] == row['name_of_city']]['state_name'].to_numpy()[0],

                                'latitude':df_cities[df_cities['name_of_city'] == row['name_of_city']]['location'].to_numpy()[0].split(',')[0],

                                'longitude':df_cities[df_cities['name_of_city'] == row['name_of_city']]['location'].to_numpy()[0].split(',')[1]

                                },

                                ignore_index=True)

df_cities_stations.describe()
df_cities_stations.hist(bins = 10 , figsize= (12,16))
fig, axs = plt.subplots(1,2)

axs[0].scatter(df_cities_stations['nb_stations'],df_cities_stations['population'])

axs[0].set_xlabel('Number of stations')

axs[0].set_ylabel('Population')

axs[1].scatter(df_cities_stations['nb_trains'],df_cities_stations['population'])

axs[1].set_xlabel('Number of trains')

plt.show()
many_stations = df_cities_stations[df_cities_stations['nb_stations'] >= 3]

many_stations
df_cities_stations.sort_values('nb_trains', ascending=False).head(10)
fig, ax = plt.subplots()

ax.scatter(df_cities_stations['nb_start_trains'],df_cities_stations['nb_end_trains'])

ax.set_xlabel('Number of starting trains')

ax.set_ylabel('Number of ending trains')

plt.show()
fig, axs = plt.subplots(3,1,figsize=(12,12))

axs[0].scatter(df_cities_stations['literacy'],df_cities_stations['nb_trains'])

axs[0].set_ylabel('Number of trains')

axs[0].set_xlabel('Literacy')

axs[1].scatter(df_cities_stations['literacy_gap'],df_cities_stations['nb_trains'])

axs[1].set_ylabel('Number of trains')

axs[1].set_xlabel('Gender inequality against literacy')

axs[2].scatter(df_cities_stations['graduate']/df_cities_stations['population'],df_cities_stations['nb_trains'])

axs[2].set_ylabel('Number of trains')

axs[2].set_xlabel('Rate of graduated inhabitants')

plt.show()
states = set(df_stations['state'])

stations_by_state = {}

for s in states:

    stations_by_state[s] = []

    for idx,row in df_stations.iterrows():

        if row['state'] == s:

            #stations_by_state[s].append(row['city'])

            c = row['city']

            for C in df_cities['name_of_city']:

                if c in C.lower() or C.lower() in c:

                    S = df_cities[df_cities['name_of_city'] == C]['state_name'].to_numpy()[0]

                    stations_by_state[s].append(S)

            

stations_by_state
stations_by_state.pop(8)

stations_by_state.pop('02')

stations_by_state.pop('04')

stations_by_state.pop('06')

stations_by_state.pop('07')

trad_table = {}

for sbs in stations_by_state:

    threshold = 2 * len(stations_by_state[sbs]) / 3

    sts = set(stations_by_state[sbs])

    aux = {}

    for s in sts:

        aux[s] = stations_by_state[sbs].count(s)

    trad_table[sbs] = ''

    for t in sts:

        if aux[t] >= threshold:

            trad_table[sbs] = t

            

trad_table