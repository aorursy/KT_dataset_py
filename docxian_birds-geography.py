import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import folium
df = pd.read_csv('../input/birdsong-recognition/train.csv')
df.head()
df.shape
df.columns
# country distribution

df.country.value_counts()
df.country.value_counts()[0:10].plot(kind='bar')

plt.grid()

plt.show()
# convert lat/lon to numeric/NaN

df.latitude = pd.to_numeric(df.latitude, errors='coerce')

df.longitude = pd.to_numeric(df.longitude, errors='coerce')
# first simple plot of locations

sns.scatterplot(x='longitude', y='latitude', data=df)

plt.grid()

plt.show()
# eval frequencies

bird_freq = df.ebird_code.value_counts()

bird_freq
# select first 10 categories and plot in color

df_select = df[df.ebird_code.isin(bird_freq[0:9+1].index)]

sns.scatterplot(x='longitude', y='latitude', hue='ebird_code', data=df_select, palette='colorblind')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # move legend out of the plot area



plt.grid()

plt.show()
# select next 10 categories and plot in color

df_select = df[df.ebird_code.isin(bird_freq[10:19+1].index)]

sns.scatterplot(x='longitude', y='latitude', hue='ebird_code', data=df_select, palette='colorblind')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.grid()

plt.show()
# select next 10 categories and plot in color

df_select = df[df.ebird_code.isin(bird_freq[20:29+1].index)]

sns.scatterplot(x='longitude', y='latitude', hue='ebird_code', data=df_select, palette='colorblind')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.grid()

plt.show()
# select next 10 categories and plot in color

df_select = df[df.ebird_code.isin(bird_freq[30:39+1].index)]

sns.scatterplot(x='longitude', y='latitude', hue='ebird_code', data=df_select, palette='colorblind')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.grid()

plt.show()
my_bird = 'houspa'

df_example = df[df.ebird_code.isin([my_bird])]

df_example.shape
# check for missing coordinates

df_example.latitude.isna().sum()
# check for missing coordinates

df_example.longitude.isna().sum()
# remove rows with missings

df_example = df_example.dropna(axis=0, subset=['latitude','longitude'])
df_example.shape
# interactive map

zoom_factor = 2

my_map_1 = folium.Map(location=[0,0], zoom_start=zoom_factor)



for i in range(0,df_example.shape[0]):

   folium.Circle(

      location=[df_example.iloc[i]['latitude'], df_example.iloc[i]['longitude']],

      radius=np.sqrt(df_example.iloc[i]['rating'])*25000,

      color='blue',

      popup='ID: ' + str(df_example.iloc[i]['xc_id']),

      fill=True,

      fill_color='blue',

   ).add_to(my_map_1)



my_map_1 # display
my_bird = 'greegr'

df_example = df[df.ebird_code.isin([my_bird])]

df_example = df_example.dropna(axis=0, subset=['latitude','longitude'])



# interactive map

zoom_factor = 2

my_map_2 = folium.Map(location=[0,0], zoom_start=zoom_factor)



for i in range(0,df_example.shape[0]):

   folium.Circle(

      location=[df_example.iloc[i]['latitude'], df_example.iloc[i]['longitude']],

      radius=np.sqrt(df_example.iloc[i]['rating'])*25000,

      color='blue',

      popup='ID: ' + str(df_example.iloc[i]['xc_id']),

      fill=True,

      fill_color='blue',

   ).add_to(my_map_2)



my_map_2 # display