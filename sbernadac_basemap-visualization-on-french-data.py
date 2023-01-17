# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.colors as colors

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#https://data.iledefrance.fr/explore/dataset/entreprises-immatriculees-en-2017/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true

creation = pd.read_csv("../input/entreprises-immatriculees-en-2017.csv",sep=';')

#https://data.iledefrance.fr/explore/dataset/entreprises-radiees-en-2017/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true

deletion = pd.read_csv("../input/entreprises-radiees-en-2017.csv",sep=';')

#https://data.iledefrance.fr/explore/dataset/base-comparateur-de-territoires/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true

base = pd.read_csv("../input/base-comparateur-de-territoires.csv",sep=';')

#https://datanova.legroupe.laposte.fr/explore/dataset/laposte_hexasmal/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true

code = pd.read_csv("../input/laposte_hexasmal.csv",sep=';')

#we use code to replace insee codes by zip codes

#check if all insee codes from base are in code dataframe

base[~base['CODGEO'].isin(code['Code_commune_INSEE'])]['CODGEO']

#result : 77166 77170 7729 but no data about firms linked with those codes:

#base[base['CODGEO']==77166]['ETTOT15'] == 0

#base[base['CODGEO']==77170]['ETTOT15'] == 0

#base[base['CODGEO']==77299]['ETTOT15'] == 0



#we need to get rid of incorrect rows. Some of rows contain non numeric characters. We will select only those values that can be transformed into numbers.

code=code[code['Code_commune_INSEE'].apply(lambda x: str(x).isdigit())]

code['Code_commune_INSEE']=code['Code_commune_INSEE'].astype(int)



#some rows dont contain geographicel parameters:

base=base[base['geo_point_2d'].apply(lambda x: ',' in str(x))]



#we add zip codes to base dataframe accordingly to insee codes

base=base.merge(code, how='left', left_on='CODGEO', right_on='Code_commune_INSEE')





#now we can compute creation and deletion number by zip codes

creation['creation_count'] = creation.groupby('Code postal')['Code postal'].transform('count')

deletion['deletion_count'] = deletion.groupby('Code postal')['Code postal'].transform('count')

creation=creation[['creation_count','Code postal','Ville']].drop_duplicates(subset=['Code postal'])

deletion=deletion[['deletion_count','Code postal']].drop_duplicates(subset=['Code postal'])



#now we can add deletion and creation table accordingly to zip codes

base=base.merge(creation, how='left', left_on='Code_postal', right_on='Code postal')

base=base.merge(deletion, how='left', left_on='Code_postal', right_on='Code postal')



#replace nan values for creation_count and deletion_count by 0

base.fillna(0, inplace=True)



#drop Ville with no name:

base.drop(base[base['Ville']==0].index,inplace=True)



#just keep relevant columns:

base=base[['CODGEO','Ville','P14_POP','ETTOT15','creation_count','deletion_count','geo_point_2d']]



base["longitude"] = base['geo_point_2d'].apply(lambda x: str(x).split(',')[0]).astype(float)

base["latitude"] = base['geo_point_2d'].apply(lambda x: str(x).split(',')[1]).astype(float)



#use Paris to center map

lon=base[base['CODGEO']==75101]['longitude']

lat=base[base['CODGEO']==75101]['latitude']
#compute size to apply differents colors

base['size']=(base['ETTOT15']).astype(float)

lats = base["longitude"].values.tolist()

lons = base["latitude"].values.tolist()

size = base['size'].values.tolist()
#compute cities which have more firms creation than deletion 

base_up=base[base['creation_count']-base['deletion_count']>50]

lats_up = base_up["longitude"].values.tolist()

lons_up = base_up["latitude"].values.tolist()

size_up = base_up['size'].values.tolist()
#compute cities which have more firms deletion than creation 

base_down=base[base['creation_count']-base['deletion_count']<-5]

lats_down = base_down["longitude"].values.tolist()

lons_down = base_down["latitude"].values.tolist()

size_down = base_down['size'].values.tolist()
#get main cities by number of firms

base['full']=base.groupby('Ville')['ETTOT15'].transform('sum')

base=base[['longitude','latitude','Ville','ETTOT15']].drop_duplicates(subset=['Ville'])

main_cities = base.sort_values(by=["ETTOT15"], ascending=False).head(10)

#main_cities = main_cities.iloc[50:200]

main_cities_names = main_cities["Ville"].values.tolist()

main_cities_lats = main_cities["longitude"].values.tolist()

main_cities_lons = main_cities["latitude"].values.tolist()
# Creating new plot

plt.figure(figsize=(20,20))

# Load map of France

map = Basemap(projection='lcc', 

            lat_0=lat,

            lon_0=lon,

            resolution='h',

            llcrnrlon=1.5, llcrnrlat=48.2,

            urcrnrlon=3.51, urcrnrlat=49.2)



# Draw parallels.

parallels = np.arange(48.,49.2,.2)

map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

# Draw meridians

meridians = np.arange(0.,4.,.5)

map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)



map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers(color='lightblue')

#map.fillcontinents(color='beige')

#map.drawmapscale( 1.6, 48.1,lon, lat, length=10)

map.drawlsmask()



# Draw scatter plot with all cities number of firms

x,y = map(lons, lats)

map.scatter(x, y, s=size, alpha=0.6, c=size, norm=colors.LogNorm(vmin=1, vmax=max(size)), cmap='viridis')

cbar = map.colorbar(location="bottom", pad="4%")

cbar.set_label('Number of Firms by cities',fontsize=18)



# Draw scatter plot of cities with more creation than deletion in 2017

x1, y1 = map(lons_up, lats_up)

map.scatter(x1, y1, c="blue")



# Draw scatter plot of cities with more creation than deletion in 2017

x2, y2 = map(lons_down, lats_down)

map.scatter(x2, y2, c="red")



plt.annotate('10 cities with the highiest number of firms',xy=(map(2.5,49.12)), fontsize=24)

for i in range(len(main_cities)):

    plt.annotate(main_cities_names[i], xy=(map(main_cities_lons[i],  main_cities_lats[i])), fontsize=8)



    

plt.title("Paris state firms number in 2015", fontsize=40, fontweight='bold',y=1.02)   

bl = mpatches.Patch(color='blue', label='cities with more than 50 new firms in 2017')

red = mpatches.Patch(color='red', label='cities with more than 5 deleted firms in 2017')



plt.legend(handles=[bl, red], ncol=2, frameon=True, fontsize=18,handlelength=1, loc = 8, borderpad = 1.8,handletextpad=1)

plt.show()
#import gmplot



#base['dif']=base['creation_count']-base['deletion_count']

#gmap = gmplot.GoogleMapPlotter(lon, lat, 9)



#for row in base.index:

        #if base.loc[row,['dif']].values<0:

                #gmap.circle(base.loc[row,['longitude']], base.loc[row,["latitude"]],c='red', radius=base.loc[row,['size']].values)

        #elif base.loc[row,['dif']].values==0:

                #gmap.circle(base.loc[row,['longitude']], base.loc[row,["latitude"]],c='black', radius=base.loc[row,['size']].values)

        #else:

                #gmap.circle(base.loc[row,['longitude']], base.loc[row,["latitude"]],c='green', radius=base.loc[row,['size']].values)



#gmap.draw('mytest.html')
