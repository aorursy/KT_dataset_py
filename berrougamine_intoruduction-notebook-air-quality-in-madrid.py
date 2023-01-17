%matplotlib  inline



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import glob

import missingno as msno





import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
colors = ["windows blue", "amber", "faded green", "dusty purple"]

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 

            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })
path =r'../input/csvs_per_year/csvs_per_year/' 

allFiles = glob.glob(path + "/*.csv")

frame = pd.DataFrame()

list_ = []

#Boucle pour rassembler tous les fic csv dans frame

for file_ in allFiles:

    df = pd.read_csv(file_,index_col=None, header=0)

    list_.append(df)

frame = pd.concat(list_)



cols = ['date', 'station', 'BEN', 'CH4', 'CO', 'EBE', 'MXY', 'NMHC', 'NO', 'NO_2', 'NOx', 'OXY',

       'O_3', 'PM10', 'PM25', 'PXY', 'SO_2', 'TCH', 'TOL']

frame = frame[cols]

frame = frame.sort_values(['station', 'date'])
frame.tail(5)
frame.info()
frame.describe()
#Les valeurs manquantes dans data frame

msno.matrix(frame);
msno.bar(frame);
#les valeurs manquantes en chiffres

frame.isnull().sum()
stations = pd.read_csv('../input/stations.csv')

stations.head()
locations  = stations[['lat', 'lon']]

locationlist = locations.values.tolist()



popup = stations[['name']]



import folium

map_osm = folium.Map(location=[40.44, -3.69],

                    # tiles='Stamen Toner',

                     zoom_start=11) 



for point in range(0, len(locationlist)):

    folium.Marker(locationlist[point], popup=popup.iloc[point,0]).add_to(map_osm)

    

map_osm
msno.matrix(stations);
stations.isnull().sum()
stations.head(10)