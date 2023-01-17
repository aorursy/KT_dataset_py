import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import folium

from folium.plugins import HeatMap



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read file

df = pd.read_csv('../input/global-significant-earthquake-database-from-2150bc/Worldwide-Earthquake-database.csv')
df.head()
# check dimensions

df.shape
# show all column names

df.columns
zoom_factor = 1.1 # setting for maps
plt.figure(figsize=(16,4))

plt.scatter(df.YEAR, df.EQ_MAG_MW, c='#00006040')

plt.xlabel('Year')

plt.ylabel('Magnitude (EQ_MAG_MW)')

plt.grid()

plt.show()
plt.figure(figsize=(16,4))

plt.scatter(df.YEAR, df.INTENSITY, c='#00006040')

plt.xlabel('Year')

plt.ylabel('Intensity')

plt.grid()

plt.show()
# zoom in

plt.figure(figsize=(16,4))

plt.scatter(df.YEAR, df.INTENSITY)

plt.xlabel('Year')

plt.ylabel('Intensity')

plt.xlim(1900,2020)

plt.grid()

plt.show()
# same for magnitude

plt.figure(figsize=(16,4))

plt.scatter(df.YEAR, df.EQ_MAG_MW)

plt.xlabel('Year')

plt.ylabel('Magnitude (EQ_MAG_MW)')

plt.xlim(1900,2020)

plt.grid()

plt.show()
# same for magnitude

plt.figure(figsize=(16,4))

plt.scatter(df.YEAR, df.EQ_MAG_MW, c='#00006040')

plt.xlabel('Year')

plt.ylabel('Magnitude (EQ_MAG_MW)')

plt.xlim(1990,2020)

plt.grid()

plt.show()
# filter by magnitude

df_select_8 = df.loc[df.EQ_MAG_MW > 8].copy()
# plot

plt.figure(figsize=(16,4))

plt.scatter(df_select_8.YEAR, df_select_8.EQ_MAG_MW)

plt.xlabel('Year')

plt.ylabel('Magnitude (EQ_MAG_MW)')

plt.title('Most significant EQs since 1900')

plt.xlim(1900,2020)

plt.grid()

plt.show()
# convert lat/lon to numeric

df_select_8.loc[:,'LATITUDE'] = pd.to_numeric(df_select_8['LATITUDE'], downcast='float')

df_select_8.loc[:,'LONGITUDE'] = pd.to_numeric(df_select_8['LONGITUDE'], downcast='float')
sel_columns = ['I_D','YEAR','EQ_MAG_MW','LATITUDE','LONGITUDE']

df_select_8 = df_select_8[sel_columns]

df_select_8
df_select_8.shape
# interactive map

my_map_1 = folium.Map(location=[0,0], zoom_start=zoom_factor)



for i in range(0,df_select_8.shape[0]):

   folium.Circle(

      location=[df_select_8.iloc[i]['LATITUDE'], df_select_8.iloc[i]['LONGITUDE']],

      radius=np.sqrt(df_select_8.iloc[i]['EQ_MAG_MW']-8)*200000,

      color='red',

      popup='ID:' + str(int(df_select_8.iloc[i]['I_D'])) + '- YEAR:' + str(int(df_select_8.iloc[i]['YEAR'])) + '- Magnitude:' 

       + str(df_select_8.iloc[i]['EQ_MAG_MW']),

      fill=True,

      fill_color='red'

   ).add_to(my_map_1)



my_map_1 # display
# filter by magnitude

df_select_6 = df.loc[df.EQ_MAG_MW > 6].copy()



# convert lat/lon to numeric

df_select_6.loc[:,'LATITUDE'] = pd.to_numeric(df_select_6.LATITUDE, downcast='float')

df_select_6.loc[:,'LONGITUDE'] = pd.to_numeric(df_select_6.LONGITUDE, downcast='float')
sel_columns = ['I_D','YEAR','EQ_MAG_MW','LATITUDE','LONGITUDE']

df_select_6 = df_select_6[sel_columns]

df_select_6.shape
df_select_6
# use heatmap to display many earthquakes

my_map_2 = folium.Map(location=[0,0], zoom_start=zoom_factor)

HeatMap(data=df_select_6[['LATITUDE', 'LONGITUDE']], radius=10).add_to(my_map_2)



my_map_2 # display
my_map_3 = my_map_1



HeatMap(data=df_select_6[['LATITUDE', 'LONGITUDE']], radius=10).add_to(my_map_3)



my_map_3
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# convert to string first

df.LOCATION_NAME = df.LOCATION_NAME.astype('str')



stopwords = set(STOPWORDS)

text = " ".join(txt for txt in df['LOCATION_NAME'])
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()