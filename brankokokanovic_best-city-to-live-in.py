import pandas as pd

df=pd.read_json(r'../input/wiki-climate.json', encoding='utf-8')
df = df[['name', 'population', 'country', 'gps_lat', 'gps_lon', 'location', 'year mean C', 'year mean C stdev']]
df[['population', 'gps_lat', 'gps_lon', 'year mean C','year mean C stdev']] = \
    df[['population', 'gps_lat', 'gps_lon', 'year mean C','year mean C stdev']].apply(pd.to_numeric)
df=df.dropna()
import folium
import matplotlib.pyplot as plt
import matplotlib

norm = matplotlib.colors.Normalize(vmin=df['year mean C'].min(), vmax=df['year mean C'].max())
cw=plt.get_cmap('coolwarm')

m = folium.Map(location=[0,0], zoom_start=2)
for _, row in df.iterrows():
    color = matplotlib.colors.rgb2hex(cw(norm(row['year mean C']))[:3]),
    folium.Circle(
        radius = 4000,
        location = [row['gps_lat'], row['gps_lon']],
        color = color,
        fill = True,
        fill_opacity = 1,
  ).add_to(m)
m
norm = matplotlib.colors.Normalize(vmin=df['year mean C stdev'].min(), vmax=df['year mean C stdev'].max())
cw=plt.get_cmap('Greys')

m = folium.Map(location=[0,0], zoom_start=2)
for _, row in df.iterrows():
    color = matplotlib.colors.rgb2hex(cw(norm(row['year mean C stdev']))[:3]),
    folium.Circle(
        radius = 4000,
        location = [row['gps_lat'], row['gps_lon']],
        color = color,
        fill = True,
        fill_opacity = 1,
  ).add_to(m)
m
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(df['year mean C'], df['year mean C stdev'])
plt.xlabel('Mean temp [°C]')
plt.ylabel('Mean temp stdev')
plt.show()
m,b = np.polyfit(df['year mean C'], df['year mean C stdev'], 1)
df['residue_stdev']=df['year mean C stdev'] - (m*df['year mean C']+b)
display(df[(df.population>500000)][['name', 'residue_stdev']].sort_values('residue_stdev')[0:10])
display(df[(df.population>500000)][['name', 'residue_stdev']].sort_values('residue_stdev', ascending=False)[0:10])
plt.scatter(df['gps_lat'], df['year mean C'])
plt.xlabel('Latitude [°]')
plt.ylabel('Mean temp [C]')
plt.show()
plt.scatter(df['gps_lat'].abs(), df['year mean C'])
plt.xlabel('Latitude (absolute) [°]')
plt.ylabel('Mean temp [C]')
plt.show()
z = np.polyfit(df['gps_lat'].abs(), df['year mean C'], 3)
p = np.poly1d(z)
df['residue_lat']=df['year mean C'] - p(df['gps_lat'])

plt.scatter(df['gps_lat'].abs(), df['year mean C'])
plt.plot(df['gps_lat'].abs(), p(df['gps_lat'].abs()), '.', color='red')
plt.xlabel('Latitude (absolute) [°]')
plt.ylabel('Mean temp [C]')
plt.show()

display(df[(df.population>500000)][['name', 'residue_lat']].sort_values('residue_lat')[0:10])
display(df[(df.population>500000)][['name', 'residue_lat']].sort_values('residue_lat', ascending=False)[0:10])
display(df[(df.population>500000) & (df.gps_lat>0)][['name', 'residue_lat', 'gps_lat']].sort_values('residue_lat', ascending=False)[0:10])
from sklearn.preprocessing import normalize

df['metric'] = \
    normalize(df['residue_stdev'][:,np.newaxis], axis=0).ravel() * \
    normalize(df['year mean C'][:,np.newaxis], axis=0).ravel() 
display(df[(df.population>1000000)][['country','name', 'residue_stdev', 'year mean C', 'metric']].sort_values('metric')[0:30])