import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import folium

from folium.plugins import HeatMap



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from mlxtend.plotting import ecdf # empirical CDF plot



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read file

df = pd.read_csv('../input/california-wildfire-incidents-20132020/California_Fire_Incidents.csv')

df.head(10)
df.shape
# show all column names

df.columns
# counties (show top 10 only)

df.Counties.value_counts()[0:10]
# plot Acres Burned vs Year

plt.figure(figsize=(16,4))

plt.scatter(df.ArchiveYear, df.AcresBurned, color='blue', alpha=0.25)

plt.xlabel('Archive Year')

plt.ylabel('AcresBurned')

plt.grid()

plt.show()
# same data visualized as violin plot

plt.figure(figsize=(16,4))

sns.violinplot(x='ArchiveYear', y='AcresBurned', data=df)

plt.grid()

plt.title('Acres Burned vs Year')

plt.show()
# log trafo

df['log10AcresBurned'] = np.log10(df.AcresBurned+0.1) # add 0.1 to avoid problems with log10(0)
# violin plot in log coordinates

plt.figure(figsize=(16,4))

sns.violinplot(x='ArchiveYear', y='log10AcresBurned', data=df)

plt.grid()

plt.title('Acres Burned vs Year (log scale)')

plt.show()
# acres burned - aggregate by year

acres_sum = df.groupby(by='ArchiveYear').AcresBurned.sum()

acres_sum
plt.scatter(acres_sum.index, acres_sum)

plt.grid()

plt.title('Acres Burned sum per year')

plt.show()
# fatalities per year

fatalities_sum = df.groupby(by='ArchiveYear').Fatalities.sum()

fatalities_sum
plt.scatter(fatalities_sum.index, fatalities_sum)

plt.grid()

plt.title('Fatalities per year')

plt.show()
cond_statements = df.ConditionStatement[~df.ConditionStatement.isna()]

cond_statements
stopwords = set(STOPWORDS)

# add more context specific stopwords

stopwords.update({'www','href','http','https'})
# show wordcloud

text = " ".join(txt for txt in cond_statements)



wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# check coordinates; there are quite a few unrealistic ones

plt.scatter(df.Longitude, df.Latitude)

plt.grid()

plt.show()
# remove those rows having unrealistic coordinates

df_select = df[df.Longitude<-115]

df_select = df_select[(df_select.Latitude<44) & (df_select.Latitude > 30)]

plt.scatter(df_select.Longitude, df_select.Latitude)

plt.grid()

plt.show()
# ok still one outlier in Nevada, let's get rid of that too

outlier = df_select[df_select.CanonicalUrl=='/incidents/2013/8/6/tram-fire/']
print('Outlier Lon/Lat:', outlier.Longitude, outlier.Latitude)
df_select = df_select[df_select.CanonicalUrl!='/incidents/2013/8/6/tram-fire/']

plt.scatter(df_select.Longitude, df_select.Latitude)

plt.grid()

plt.show()
# interactive map

zoom_factor = 5 # inital map size

radius_scaling = 50 # scaling of bubbles



my_map_1 = folium.Map(location=[36,-120], zoom_start=zoom_factor)



for i in range(0,df_select.shape[0]):

   folium.Circle(

      location=[df_select.iloc[i]['Latitude'], df_select.iloc[i]['Longitude']],

      radius=np.sqrt(df_select.iloc[i]['AcresBurned'])*radius_scaling,

      color='red',

      popup='CanonicalUrl:' + df_select.iloc[i]['CanonicalUrl'] + ' - Year:' + str(int(df_select.iloc[i]['ArchiveYear'])) + ' - Acres Burned:' 

      + str(df_select.iloc[i]['AcresBurned']),

      fill=True,

      fill_color='red'

   ).add_to(my_map_1)



my_map_1 # display
# use heatmap

my_map_2 = folium.Map(location=[36,-120], zoom_start=zoom_factor)

HeatMap(data=df_select[['Latitude', 'Longitude']], radius=10).add_to(my_map_2)



my_map_2 # display
# select year 2018

df_select_2018 = df_select[df_select.ArchiveYear==2018]
# interactive map

zoom_factor = 5 # inital map size

radius_scaling = 50 # scaling of bubbles



my_map_2 = folium.Map(location=[36,-120], zoom_start=zoom_factor)



for i in range(0,df_select_2018.shape[0]):

   folium.Circle(

      location=[df_select_2018.iloc[i]['Latitude'], df_select_2018.iloc[i]['Longitude']],

      radius=np.sqrt(df_select_2018.iloc[i]['AcresBurned'])*radius_scaling,

      color='red',

      popup='CanonicalUrl:' + df_select_2018.iloc[i]['CanonicalUrl'] + ' - Year:' + str(int(df_select_2018.iloc[i]['ArchiveYear'])) + ' - Acres Burned:' 

      + str(df_select_2018.iloc[i]['AcresBurned']),

      fill=True,

      fill_color='red'

   ).add_to(my_map_2)



my_map_2 # display
ecdf(df_select_2018.AcresBurned)

plt.grid()

plt.title('2018 - Acres Burned')

plt.show()
ecdf(np.log10(df_select_2018.AcresBurned))

plt.grid()

plt.title('2018 - Acres Burned - Log Scale')

plt.show()