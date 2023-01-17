# data analysis

import numpy as np 

import pandas as pd

import geopandas as gpd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import folium
# import data

df = pd.read_csv('../input/crimeanalysis/crime_by_state.csv')

df.sample(5)
# last rows

df.tail()
# shape

df.shape
# columns 

df.columns
# df.dtypes
# df.info()
# df.describe(include='all')
# missing values

df.isna().sum()
# unique values

df.nunique()
# df['STATE/UT'].value_counts()

df['STATE/UT'].unique()
# df['Year'].value_counts()

df['Year'].unique()
df['STATE/UT'] = df['STATE/UT'].str.title()
# numerical cols

cols = ['Murder', 'Assault on women', 'Kidnapping and Abduction', 'Dacoity', 

        'Robbery', 'Arson', 'Hurt', 'Prevention of atrocities (POA) Act', 

        'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']



# total column

df['Total'] = df[cols].sum(axis=1)
# drop columns with aggregate values

pattern = "Total*"

fltr = df['STATE/UT'].str.contains(pattern)

df = df[~fltr]

# df.tail()
# df['STATE/UT'].value_counts()

df['STATE/UT'].unique()
plt.figure(figsize=(10, 10))

df_pivot = df[['STATE/UT', 'Year', 'Total']].pivot_table(values='Total', index='STATE/UT', columns='Year', aggfunc='sum')

sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='RdPu')

plt.suptitle('Total cases over the years in each States')

plt.xlabel('')

plt.ylabel('')

plt.show()
plt.figure(figsize=(40, 15))

df_pivot = df.groupby('STATE/UT')[cols].sum()

sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='GnBu')

plt.suptitle('Number of each cases in each states')

plt.xlabel('')

plt.ylabel('')

plt.show()
# plt.figure(figsize=(8, 10))

# df_pivot = df[['STATE/UT', 'Year', 'Murder']].pivot_table(values='Murder', index='STATE/UT', columns='Year', aggfunc='sum')

# sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='Greens')

# plt.suptitle('Murder')

# plt.xlabel('')

# plt.ylabel('')

# plt.show()
for i in cols:

    plt.figure(figsize=(8, 10))

    df_pivot = df[['STATE/UT', 'Year', i]].pivot_table(values=i, index='STATE/UT', columns='Year', aggfunc='sum')

    sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='Greens')

    plt.suptitle(i)

    plt.xlabel('')

    plt.ylabel('')

    plt.show()
# plt.figure(figsize=(8, 20))

# sns.barplot(y='STATE/UT', x='Murder', data=df)

# plt.show()
# state_murder = pd.DataFrame(df.groupby(['STATE/UT','Year'])['Murder'].sum())

# state_murder = state_murder.sort_values('Murder', ascending=False)

# state_murder.reset_index(inplace=True)

# state_murder.head()
# plt.figure(figsize=(30, 12))

# sns.lineplot(x="Year", y="Murder", hue="STATE/UT", data=state_murder)

# plt.show()
# import district level shape files

dist_gdf = gpd.read_file('../input/india-district-wise-shape-files/output.shp')



# group by state

states_gdf = dist_gdf.dissolve(by='statename').reset_index() 



# just select statename and geometry column

states_gdf = states_gdf[['statename', 'geometry']]
# replace state's name

states_gdf['statename'] = states_gdf['statename'].replace('Ladakh', 'Jammu & Kashmir')

states_gdf['statename'] = states_gdf['statename'].replace('Telangana', 'Andhra Pradesh')

states_gdf['statename'] = states_gdf['statename'].replace('Andaman & Nicobar Islands', 'A & N Islands')

states_gdf['statename'] = states_gdf['statename'].replace('Chhatisgarh', 'Chhattisgarh')

states_gdf['statename'] = states_gdf['statename'].replace('Dadra & Nagar Haveli', 'D & N Haveli')

states_gdf['statename'] = states_gdf['statename'].replace('Orissa', 'Odisha')

states_gdf['statename'] = states_gdf['statename'].replace('Pondicherry', 'Puducherry')

states_gdf['statename'] = states_gdf['statename'].replace('NCT of Delhi', 'Delhi')



# group 10 years of data

states_df = df.groupby('STATE/UT')[cols].sum().reset_index()

states_df.head()
# merge shape file with count file

states_full = pd.merge(states_gdf, states_df, left_on='statename', right_on='STATE/UT', how='left')

states_full.head()
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

fig.suptitle('No. of Hate crimes from 2001-2012', fontsize=16)

cmap = 'bwr'



states_full.plot(column='Murder', ax=axes[0,0], cmap=cmap)   

axes[0,0].set_title('Murder')

axes[0,0].set_axis_off()                                       



states_full.plot(column='Assault on women', ax=axes[0,1], cmap=cmap)   

axes[0,1].set_title('Assault on women')

axes[0,1].set_axis_off()          



states_full.plot(column='Kidnapping and Abduction', ax=axes[0,2], cmap=cmap)   

axes[0,2].set_title('Kidnapping and Abduction')

axes[0,2].set_axis_off()          



states_full.plot(column='Dacoity', ax=axes[0, 3], cmap=cmap)   

axes[0, 3].set_title('Dacoity')

axes[0, 3].set_axis_off()          



states_full.plot(column='Robbery', ax=axes[1,0], cmap=cmap)   

axes[1,0].set_title('Robbery')

axes[1,0].set_axis_off()          



states_full.plot(column='Arson', ax=axes[1,1], cmap=cmap)   

axes[1,1].set_title('Arson')

axes[1,1].set_axis_off()    



states_full.plot(column='Hurt', ax=axes[1,2], cmap=cmap)   

axes[1,2].set_title('Hurt')

axes[1,2].set_axis_off()          



states_full.plot(column='Prevention of atrocities (POA) Act', ax=axes[1,3], cmap=cmap)   

axes[1,3].set_title('Prevention of atrocities (POA) Act')

axes[1,3].set_axis_off()          



states_full.plot(column='Protection of Civil Rights (PCR) Act', ax=axes[2,0], cmap=cmap)   

axes[2,0].set_title('Protection of Civil Rights (PCR) Act')

axes[2,0].set_axis_off()  



states_full.plot(column='Other Crimes Against SCs', ax=axes[2,1], cmap=cmap)   

axes[2,1].set_title('Other Crimes Against SCs')

axes[2,1].set_axis_off()  



axes[2,2].set_axis_off()  



axes[2,3].set_axis_off()  



plt.show()
# states_df.head(40)
# ind_dist_map.head(50)
# m = folium.Map(location=[23, 78.9629], tiles='cartodbpositron',

#                min_zoom=4, max_zoom=6, zoom_start=4)



# folium.Choropleth(states_full, data=states_full, 

#                   key_on='feature.properties.statename',

#                   columns=['statename', 'Murder'], 

#                   fill_color='YlGnBu',

#                   line_weight=0.1,

#                   line_opacity=0.5,

#                   legend_name='No. of reported cases').add_to(m)



# folium.LayerControl().add_to(m)



# m