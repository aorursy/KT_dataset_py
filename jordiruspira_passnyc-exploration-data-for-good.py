import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/2016 School Explorer.csv')
registrations = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')

df.head()
print('----- Dataframe Information -----')
print('---------------------------------')
print(df.info())
print('----- Registrations Information -----')
print('-------------------------------------')
print(registrations.info())
for i in df.columns:
    print(i,',' ,'{:.1%}'.format(np.mean(df[i].isnull())),'nulls',',',type(df[i][0]), 
    df[i].nunique(), 'unique values')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('$', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace(',', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('.', '')
df['School Income Estimate'] = df['School Income Estimate'].astype(float)

df = df.drop(columns=['Adjusted Grade', 'New?', 'Other Location Code in LCGMS'])
corrcolumns = ['SED Code', 'District', 'Latitude', 'Longitude', 'Zip', 'Economic Need Index'] # Community School? would be interesting aswell
for i in corrcolumns:
    print('The correlation value between School Income Estimate and ', i, 'is: ', df['School Income Estimate'].corr(df[i]))

registrations.head()
for i in registrations.columns:
    print(i,',' ,'{:.1%}'.format(np.mean(registrations[i].isnull())),'nulls',',',type(registrations[i][0]), 
    registrations[i].nunique(), 'unique values')
print('Total number of students who registered for the SHSAT:',np.sum(registrations['Number of students who registered for the SHSAT']))
print('Total number of students who took the SHSAT:', np.sum(registrations['Number of students who took the SHSAT']))
registrations['% of taken over registered students'] = (registrations['Number of students who took the SHSAT'])*100/(registrations['Number of students who registered for the SHSAT'])

registrations.head()
df['District'].unique()
manh_districts = [1,2,3,4,5,6]
bronx_districts = [7, 8,9,10,11,12]
brook_districts = [13,14,15,16,17,18,19,20,21,22,23,32]
queens_districts = [24,25,26,27,28,29,30]
staten_districts = [31]


df.loc[df['District'].isin(manh_districts), 'Borough'] = 'Manhattan'
df.loc[df['District'].isin(bronx_districts), 'Borough'] = 'Bronx'
df.loc[df['District'].isin(brook_districts), 'Borough'] = 'Brooklyn'
df.loc[df['District'].isin(queens_districts), 'Borough'] = 'Queens'
df.loc[df['District'].isin(staten_districts), 'Borough'] = 'Staten Island'
sns.pairplot(df[['Borough','Latitude','Longitude','Economic Need Index']], kind="scatter", hue="Borough", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

df.loc[:,'Percent Black'] = df.loc[:,'Percent Black'].str.replace('%', '')
df.loc[:,'Percent Black'] = df.loc[:,'Percent Black'].astype(float)
df.loc[:,'Percent Hispanic'] = df.loc[:,'Percent Hispanic'].str.replace('%', '')
df.loc[:,'Percent Hispanic'] = df.loc[:,'Percent Hispanic'].astype(float)
df.loc[:,'Percent Asian']  = df.loc[:,'Percent Asian'].str.replace('%', '')
df.loc[:,'Percent Asian']  = df.loc[:,'Percent Asian'].astype(float)
df.loc[:,'Percent White'] = df.loc[:,'Percent White'] .str.replace('%', '')
df.loc[:,'Percent White'] = df.loc[:,'Percent White'] .astype(float)


sns.lmplot('Economic Need Index', 'Percent Black', data=df, hue='Borough', fit_reg=True, size = 15)
plt.title('Percent of Black students in the different Boroughs of NYC')

sns.lmplot('Economic Need Index', 'Percent Hispanic', data=df, hue='Borough', fit_reg=True,  size = 15)
plt.title('Percent of Hispanic students in the different Boroughs of NYC')

sns.lmplot('Economic Need Index', 'Percent Asian', data=df, hue='Borough', fit_reg=True,  size = 15)
plt.title('Percent of Asian students in the different Boroughs of NYC')

sns.lmplot('Economic Need Index', 'Percent White', data=df, hue='Borough', fit_reg=True,  size = 15)
plt.title('Percent of WHite students in the different Boroughs of NYC')

plt.figure(figsize=(10,10))
ax = sns.countplot(df['District'],label="Count", order = df['District'].value_counts().index)

df_manha = df[(df['District'].isin(manh_districts))]
df_bronx = df[(df['District'].isin(bronx_districts))]
df_brook = df[(df['District'].isin(brook_districts))]
df_queens = df[(df['District'].isin(queens_districts))]
df_staten = df[(df['District'].isin(staten_districts))]

median = df['School Income Estimate'].median()
median_manha = df_manha['School Income Estimate'].median()
median_bronx = df_bronx['School Income Estimate'].median()
median_brook = df_brook['School Income Estimate'].median()
median_queens = df_queens['School Income Estimate'].median()
median_staten = df_staten['School Income Estimate'].median()
print('Global median:', median)
print('Median school income in Manhattan: $', median_manha)
print('Median school income in The Bronx: $', median_bronx)
print('Median school income in Brooklyn: $', median_brook)
print('Median school income in Queens: $', median_queens)
print('Median school income in Staten Island: $', median_staten)
import folium
from folium import plugins
from io import StringIO
import folium 

df_manha['Economic Need Index'] = df_manha['Economic Need Index'].fillna((df_manha['Economic Need Index'].mean()))


#colors = ['red', 'yellow', 'dusty purple', 'blue', 'white', 'brown', 'green', 'purple', 'orange', 'grey', 'coral']
colors = ['chartreuse', 'limegreen', 'yellowgreen', 'y', 'olive', 'indianred', 'firebrick', 'tomamto', 'orangered', 'red']
d = (df_manha['Economic Need Index']*10).astype('int')
cols = [colors[int(i)] for i in d]


map_osm2 = folium.Map([df_manha['Latitude'][0], df_manha['Longitude'][0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_manha['Latitude'], df_manha['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_bronx['Economic Need Index'] = df_bronx['Economic Need Index'].fillna((df_bronx['Economic Need Index'].mean()))


#colors = ['red', 'yellow', 'dusty purple', 'blue', 'white', 'brown', 'green', 'purple', 'orange', 'grey', 'coral']
colors = ['chartreuse', 'limegreen', 'yellowgreen', 'y', 'olive', 'indianred', 'firebrick', 'tomamto', 'orangered', 'red']
d = (df_bronx['Economic Need Index']*10).astype('int')
cols = [colors[int(i)] for i in d]


map_osm2 = folium.Map([df_bronx['Latitude'].iloc[0], df_bronx['Longitude'].iloc[0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_bronx['Latitude'], df_bronx['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_brook['Economic Need Index'] = df_brook['Economic Need Index'].fillna((df_brook['Economic Need Index'].mean()))


#colors = ['red', 'yellow', 'dusty purple', 'blue', 'white', 'brown', 'green', 'purple', 'orange', 'grey', 'coral']
colors = ['chartreuse', 'limegreen', 'yellowgreen', 'y', 'olive', 'indianred', 'firebrick', 'tomamto', 'orangered', 'red']
d = (df_brook['Economic Need Index']*10).astype('int')
cols = [colors[int(i)] for i in d]


map_osm2 = folium.Map([df_brook['Latitude'].iloc[0], df_brook['Longitude'].iloc[0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_brook['Latitude'], df_brook['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_queens['Economic Need Index'] = df_queens['Economic Need Index'].fillna((df_queens['Economic Need Index'].mean()))


#colors = ['red', 'yellow', 'dusty purple', 'blue', 'white', 'brown', 'green', 'purple', 'orange', 'grey', 'coral']
colors = ['chartreuse', 'limegreen', 'yellowgreen', 'y', 'olive', 'indianred', 'firebrick', 'tomamto', 'orangered', 'red']
d = (df_queens['Economic Need Index']*10).astype('int')
cols = [colors[int(i)] for i in d]


map_osm2 = folium.Map([df_queens['Latitude'].iloc[0], df_queens['Longitude'].iloc[0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_queens['Latitude'], df_queens['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_staten['Economic Need Index'] = df_staten['Economic Need Index'].fillna((df_staten['Economic Need Index'].mean()))


#colors = ['red', 'yellow', 'dusty purple', 'blue', 'white', 'brown', 'green', 'purple', 'orange', 'grey', 'coral']
colors = ['chartreuse', 'limegreen', 'yellowgreen', 'y', 'olive', 'indianred', 'firebrick', 'tomamto', 'orangered', 'red']
d = (df_staten['Economic Need Index']*10).astype('int')
cols = [colors[int(i)] for i in d]


map_osm2 = folium.Map([df_staten['Latitude'].iloc[0], df_staten['Longitude'].iloc[0]], zoom_start=10.2,tiles='cartodbdark_matter')

for lat, long, col in zip(df_staten['Latitude'], df_staten['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(map_osm2)

map_osm2
df_manha.plot(kind="scatter", x="Economic Need Index", y="Percent Black", figsize=(15,9) )
plt.show()
df_manha.plot(kind="scatter", x="Economic Need Index", y="Percent Hispanic", figsize=(15,9) )
plt.show()
df_manha.plot(kind="scatter", x="Economic Need Index", y="Percent Asian", figsize=(15,9) )
plt.show()
df_manha.plot(kind="scatter", x="Economic Need Index", y="Percent White", figsize=(15,9) )
plt.show()

plt.figure(figsize=(15,10))
#ax = sns.violinplot(x="District", y="Economic Need Index", hue="Community School?", data=df_manha, palette="muted")
sns.swarmplot(x="District", y="Percent Black", hue="Community School?", data=df_manha)
plt.show()
print(df_manha.shape)

ax = sns.countplot(df_manha['Supportive Environment Rating'],label="Count")       # M = 212, B = 357
plt.show()
ax2 = sns.countplot(df_manha['Community School?'],label="Count")       # M = 212, B = 357
plt.show()

print(df_manha['Grades'].unique())
print(registrations['School name'].unique())
print(df['School Name'].unique())
df.plot(kind="scatter", x="Longitude", y="Latitude", c="Economic Need Index", cmap=plt.get_cmap("jet"),label='Schools', title='New York School Population Map',
    colorbar=True, alpha=0.4, figsize=(15,9))
plt.legend()
plt.show()

df_manha.plot(kind="scatter", x="Longitude", y="Latitude", c="Economic Need Index", cmap=plt.get_cmap("jet"),label='Schools', title='New York School Population Map',
    colorbar=True, alpha=0.4, figsize=(15,9) )
plt.legend()
plt.show()

plt.figure(figsize=(15,10))
#ax = sns.violinplot(x="District", y="Economic Need Index", hue="Community School?", data=df_manha, palette="muted")
sns.swarmplot(x="District", y="Economic Need Index", hue="Community School?", data=df_manha)
plt.show()
