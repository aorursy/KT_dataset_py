import pandas as pd

import numpy as np

from collections import Counter

import plotly.express as px
df = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_stop_data.csv', low_memory = False)

force_df = pd.read_csv('/kaggle/input/minneapolis-police-stops-and-police-violence/police_use_of_force.csv')
df.head()
df.info()
year_values = []

for i in range(len(df)):

    date = df['responseDate'][i].split(" ")[0]

    year = date.split("/")[0]

    year_values.append(year)

    

year_counts = dict(Counter(year_values))

year_counts = {'year': list(year_counts.keys()), 'count': list(year_counts.values())}

years_df = pd.DataFrame(year_counts)

years_df
fig_yearly = px.pie(years_df, values = 'count', names = 'year', title = 'Yearly Cases Distribution', hole = .5, color_discrete_sequence = px.colors.diverging.Portland)

fig_yearly.show()
problem_counts_dict = dict(Counter(df['problem']))

problem_df_dict = {'problem': list(problem_counts_dict.keys()), 'count': list(problem_counts_dict.values())}



problem_df = pd.DataFrame(problem_df_dict)

problem_df
fig_yearly = px.pie(problem_df, values = 'count', names = 'problem', title = 'Type of Cases', hole = .5, color_discrete_sequence = px.colors.sequential.Agsunset)

fig_yearly.show()
import folium

from folium.plugins import FastMarkerCluster

locations = df[['lat', 'long']]

locationlist = locations.values.tolist()
map = folium.Map(location=[44.986656, -93.258133], zoom_start=12)

FastMarkerCluster(data=list(zip(df['lat'].values, df['long'].values))).add_to(map)

map
df['race'].fillna('No Data', inplace = True)

race_counts_dict = dict(Counter(df['race']))



race_counts_dict['Unknown'] += race_counts_dict['No Data']

del race_counts_dict['No Data']



race_df_dict = {'race': list(race_counts_dict.keys()), 'count': list(race_counts_dict.values())}



race_df = pd.DataFrame(race_df_dict)

race_df
fig_race = px.pie(race_df, values = 'count', names = 'race', title = 'Distribution of Races', hole = .5, color_discrete_sequence = px.colors.diverging.Temps)

fig_race.show()
force_new = force_df[['ForceType', 'EventAge', 'TypeOfResistance', 'Is911Call']]

force_new.head()
force_counts_dict = dict(Counter(force_new['ForceType']))



force_counts_dict['Unknown'] = force_counts_dict[np.nan]

del force_counts_dict[np.nan]



force_df_dict = {'force': list(force_counts_dict.keys()), 'count': list(force_counts_dict.values())}



force_type_df = pd.DataFrame(force_df_dict)

force_type_df
fig_force = px.bar(force_type_df, x = 'force', y = 'count')

fig_force.show()
fig_age_hist = px.histogram(force_new, x = 'EventAge', nbins=10, opacity = 0.7)

fig_age_hist.show()
force_df['TypeOfResistance'].fillna('Unknown', inplace = True)

cleaned_types = []

for item in force_df['TypeOfResistance']:

    p1_item = item.strip()

    p2_item = p1_item.title()

    cleaned_types.append(p2_item)

    

force_df['TypeNew'] = cleaned_types



resistance_counts_dict = dict(Counter(force_df['TypeNew']))



resistance_counts_dict['Unspecified'] += resistance_counts_dict['Unknown']

del resistance_counts_dict['Unknown']



resistance_counts_dict['Commission Of Crime'] += resistance_counts_dict['Commission Of A Crime']

del resistance_counts_dict['Commission Of A Crime']



resistance_counts_dict['Fled In Vehicle'] += resistance_counts_dict['Fled In A Vehicle']

del resistance_counts_dict['Fled In A Vehicle']



resistance_counts_dict['Assaulting Police Horse'] += resistance_counts_dict['Assaulted Police Horse']

del resistance_counts_dict['Assaulted Police Horse']



resistance_counts_df_dict = {'type': list(resistance_counts_dict.keys()), 'count': list(resistance_counts_dict.values())}



resistance_df = pd.DataFrame(resistance_counts_df_dict)

resistance_df
fig_resistance = px.pie(resistance_df, values = 'count', names = 'type', title = 'Distribution of Resistance', hole = .5, color_discrete_sequence = px.colors.diverging.Picnic)

fig_resistance.show()
_911_counts_dict = dict(Counter(force_new['Is911Call']))



_911_counts_dict['Unspecified'] = _911_counts_dict[np.nan]

del _911_counts_dict[np.nan]



_911_df_dict = {'val': list(_911_counts_dict.keys()), 'count': list(_911_counts_dict.values())}



_911_df = pd.DataFrame(_911_df_dict)

_911_df
fig_911 = px.pie(_911_df, values = 'count', names = 'val', title = 'Distribution of 911 Calls', hole = .5, color_discrete_sequence = ['#ff4757', '#10ac84', '#2f3542'])

fig_911.show()