# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pylab import rcParams

rcParams['figure.figsize'] = (8,6)

import folium



import numpy as np  

import pandas as pd

pd.set_option("max_columns", None)



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv", encoding='ISO-8859-1', dtype = "object")

#globalterrorismdb_0718dist.csv
df.shape
df.head()
df.isnull().sum()
#df.dropna(thresh=180000,axis=1).shape
plt.subplots(figsize=(18,6))

sns.barplot(df['resolution'].value_counts()[:15].index,df['resolution'].value_counts()[:15].values,palette='inferno')

plt.xticks(rotation=45)

plt.title('Top Resolutions', size=20)

plt.show()
#df.country_txt.head(10)#.sort_values(ascending=False)
plt.figure(figsize=(15,8))

df.country_txt.value_counts().head(15).plot(kind="bar")

plt.xticks(rotation=45)
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 9



sns.barplot(x=df.country_txt.value_counts(), y=df.country_txt.value_counts().index,

            order = df.country_txt.value_counts().iloc[:25].index, orient="h")
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 9



sns.barplot(x=df.provstate.value_counts(), y=df.provstate.value_counts().index,

            order = df.provstate.value_counts().iloc[:25].index, orient="h")
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 9



sns.barplot(x=df.city.value_counts(), y=df.city.value_counts().index,

            order = df.city.value_counts().iloc[:25].index, orient="h")
df_t = df[df["country_txt"] == "France"]

df_t.head()
#df_t = df[df["country_txt"] == "France"]

plt.subplots(figsize=(16,8))

sns.barplot(x=df_t.provstate.value_counts(), y=df_t.provstate.value_counts().index,

            order = df_t.provstate.value_counts().iloc[:25].index, orient="h")
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 9



#df_t = df[df["country_txt"] == "France"]



sns.barplot(x=df_t.city.value_counts(), y=df_t.city.value_counts().index,

            order = df_t.city.value_counts().iloc[:25].index, orient="h")
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 9



#df_t = df[df["country_txt"] == "France"]



sns.barplot(x=df_t.attacktype1_txt.value_counts(), y=df_t.attacktype1_txt.value_counts().index,

            order = df_t.attacktype1_txt.value_counts().iloc[:25].index, orient="h")
plt.subplots(figsize=(12,8))

plt.xticks(rotation=40)

sns.barplot(x=df_t.weaptype1_txt.value_counts().index, y=df_t.weaptype1_txt.value_counts(),

            order = df_t.weaptype1_txt.value_counts().iloc[:12].index, orient="v")
plt.subplots(figsize=(18,8))

plt.xticks(rotation=30)

sns.barplot(x=df_t.targtype1_txt.value_counts().index, y=df_t.targtype1_txt.value_counts(),

            order = df_t.targtype1_txt.value_counts().iloc[:18].index, orient="v") 
df_t['nkill'].value_counts()
plt.figure(figsize=(12,7))



df_t.nkill.value_counts().head(8).sort_values(ascending=False).plot(kind="bar")

plt.title("Count of killed people- France", size=20)

#plt.xticks(rotation=45)
df_t['nwound'].value_counts()[:25]
plt.subplots(figsize=(18,6))

sns.barplot(df_t['nwound'].value_counts()[:15].index,df_t['nwound'].value_counts()[:15].values, palette='inferno')

plt.xticks(rotation=45)

plt.title('Count of wounded in France', size=20)

plt.show()
df_t.gname.value_counts()[:25]
plt.subplots(figsize=(12,8))

sns.barplot(df_t['gname'].value_counts()[:20],df_t['gname'].value_counts()[:20].index,palette='inferno')

plt.title('Top Terrorist Group Names', size=20)

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('iyear',data=df,palette='viridis',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities around the World according to Year')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('iyear',data=df_t,palette='viridis',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities in France according to Year')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('imonth',data=df_t,palette='viridis',edgecolor=sns.color_palette('dark',7))#RdYlGn_r

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities in France monthly', size=20)

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('alternative_txt',data=df_t,palette='viridis',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=30)

plt.title('Alternative crimes')

plt.show()
#import folium



fm = folium.Map(location=[48.86,2.32], zoom_start=9)

folium.CircleMarker(location=(48.8603634,2.3208891), popup="Paris").add_to(fm)

fm.save('Paris.html')

fm
import folium

fm = folium.Map(location=[48.8647, 2.3490], zoom_start=5.3)



folium.Marker(location=(48.856644,2.34233), popup="Paris", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.56214,2.246488), popup="Avrainville", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.798328,2.309902), popup="Bagneux", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.836582,2.23914), popup="Boulogne-Billancourt", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(47.237829,6.024054), popup="Besancon", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.296482,5.36978), popup="Marseilles", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(46.306884,4.828731), popup="Macon", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.885212,2.437678), popup="Romainville", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.416095,-3.832245), popup="Tredudon", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.604652,1.444209), popup="Toulouse", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(42.699997,9.447317), popup="Bastia", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.091463,-0.045726), popup="Lourdes", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.836699,4.360054), popup="Nimes", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.583148, 7.747882), popup="Strasbourg", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(45.764043,4.835659), popup="Lyon", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.359399,-1.766148), popup="Hendaye", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.696036,7.265592), popup="Nice", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(45.648377,0.156237), popup="Angouleme", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(47.997542,-4.097899), popup="Quimper", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(47.914709,7.537626), popup="Fessenheim", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(49.49437,0.107929), popup="Le Havre", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.483152,-1.558626), popup="Biarritz", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(50.692705,3.177847), popup="Roubaix", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(41.930607,8.742907), popup="Ajaccio", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.3533,-3.872203), popup="Brennilis", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.113475,-1.675708), popup="Rennes", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(48.218878,-4.164362), popup="Dineault", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.091463,-0.045726), popup="Lourdes", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(42.567651,8.757222), popup="Calvi", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(41.591369,9.278311), popup="Porto-Vecchio", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(41.387174,9.159269), popup="Bonifacio", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(41.857045,9.399654), popup="Solenzara", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.102976,5.878219), popup="La Seyne-Sur-Mer", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(42.55,8.75), popup="La Vaccaja", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.580418,7.125102), popup="Antibes", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.388051,-1.663055), popup="Saint-Jean-de-Luz", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(42.309409,9.149022), popup="Corte", icon=folium.Icon(color="green")).add_to(fm)

folium.Marker(location=(43.481402,-1.514699), popup="Anglet", icon=folium.Icon(color="green")).add_to(fm)





fm.save('Terrorist Events.html')

fm
#import folium 

from folium import plugins

from folium.plugins import HeatMap

df_t = df_t[df_t.country_txt=='France'].reset_index(drop=True)
df_t.city.value_counts()[:50]
def heatmaps(kind):

    fig,ax=plt.subplots(figsize=(20,15),nrows=2,ncols=2)

    for i in range(2):

        for j in range(2):

            if i==0 and j==0:

                cities = 'Paris'

            elif i==0 and j==1:

                cities = "Ajaccio"

            elif i==1 and j==0:

                cities = "Bastia"

            elif i==1 and j==0:

                cities = "Marseilles"

            else:

                cities = "Porto-Vecchio"

                

            france = df_t[df_t['city']==cities]

            france = pd.pivot_table(data=france, values=kind,index='targtype1_txt',

                                    columns='attacktype1_txt',aggfunc=np.sum,fill_value=0)

            france.columns = [i.replace('/','/\n').replace(' ','\n') for i in france.columns]



            sns.heatmap(france, annot=True,ax=ax[i,j],cmap="Blues",linewidths=1)

            ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(), rotation = 30,color='b',fontdict={'fontsize':9})

            ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(), rotation = 0,color='b')

            if j!= 1:

                ax[i,j].set_ylabel("Target Attack Type 1",fontdict={'fontweight':'bold'})

            else:

                ax[i,j].set_ylabel("")



            if i==1:

                ax[i,j].set_xlabel("Attack Type 1",fontdict={'fontweight':'bold'})

            else:

                ax[i,j].set_xlabel("")

            ax[i,j].set_title('Attack Type with Its Target in %s'%cities,fontdict = {'fontweight':'bold'})

    fig.suptitle('Top 5 Cities with High Terrorism in France',y=0.93,fontsize=22,weight='bold')

    plt.show()
france_map = folium.Map(location=[48.8647, 2.3490], 

#                        tiles = "France Terrorism",

                      zoom_start = 5)



# Add data for heatmp 

f_heatmap = df_t[['latitude','longitude']]

f_heatmap = df_t.dropna(axis=0, subset=['latitude','longitude'])

f_heatmap = [[row['latitude'],row['longitude']] for index, row in f_heatmap.iterrows()]

HeatMap(f_heatmap, radius=10).add_to(france_map)



# Plot!

france_map
df_g = df[df["country_txt"] == "Germany"]

df_t.head()
plt.figure(figsize=(15,8))

df_g.city.value_counts().head(15).plot(kind="bar")

plt.xticks(rotation=40)