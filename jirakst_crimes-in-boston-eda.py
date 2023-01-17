# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
crime = pd.read_csv('../input/crime.csv', encoding='latin-1')
crime.shape
crime.columns
crime.head(5)
 #crime = crime.fillna('Unknown')

#crime[crime["DISTRICT"].str.contains("D14")]
# Count nulls

crime.isnull().sum()
# Remove nulls/NANs (empty values)

crime = crime.drop(columns='SHOOTING')

crime = crime.dropna(axis=0)
sns.countplot(data=crime, x='YEAR')
# Keep only data from complete years (2016, 2017)

crime = crime.loc[crime['YEAR'].isin([2016,2017])]
sns.countplot(data=crime, x='MONTH')
sns.countplot(data=crime, x='DAY_OF_WEEK')
sns.countplot(data=crime, x='HOUR')
# Countplot for crime types

sns.catplot(y='OFFENSE_CODE_GROUP',

           kind='count',

            height=8, 

            aspect=1.5,

            order=crime.OFFENSE_CODE_GROUP.value_counts().index,

           data=crime)
# Top 10

crime.OFFENSE_CODE_GROUP.value_counts().head(10)
# Last 5

crime.OFFENSE_CODE_GROUP.value_counts().tail(5)
from wordcloud import WordCloud



text = []

for i in crime.OFFENSE_CODE_GROUP:

    text.append(i)#here we are adding word to text array but it's looking like this ['Larency','Homicide','Robbery']

text = ''.join(map(str, text)) #Now we make all of them like this [LarencyHomicideRobbery]



wordcloud = WordCloud(width=1600, height=800, max_font_size=300).generate(text)

plt.figure(figsize=(20,17))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
import folium

from folium.plugins import HeatMap



map_hooray = folium.Map(location=[42.361145,-71.057083],

                    zoom_start = 12, min_zoom=12) #Giving the location just write boston coordinat to google



heat_df = crime[crime['YEAR']==2017] # I take 2017 cause there is more crime against to other years

heat_df = heat_df[heat_df['OFFENSE_CODE_GROUP']=='Ballistics'] 

heat_df = heat_df[['Lat', 'Long']] #giving only latitude and longitude now in heat_df just latitude and longitude

                                        #from 2017 larceny responde



folium.CircleMarker([42.356145,-71.064083],

                    radius=50,

                    popup='Homicide',

                    color='red',

                    ).add_to(map_hooray) #Adding mark on the map but it's hard to find correct place. 

                                         #it's take to muhc time

    

    

heat_data = [[row['Lat'],row['Long']] for index, row in heat_df.iterrows()]

#We have to give latitude and longitude like this [[lat, lon],[lat, lon],[lat, lon],[lat, lon],[lat, lon]]



HeatMap(heat_data, radius=10).add_to(map_hooray) #Adding map_hooray to HeatMap

map_hooray #Plotting
crime[['OFFENSE_CODE_GROUP', 'DISTRICT']].groupby(['DISTRICT'], as_index=False).count().sort_values(by='OFFENSE_CODE_GROUP')