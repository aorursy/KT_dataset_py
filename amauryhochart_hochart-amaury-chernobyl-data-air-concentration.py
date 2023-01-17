# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # graphs pyplot

import seaborn as sns # graphs

import geopandas as gpd # map



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from sklearn.linear_model import LogisticRegression # regression logistique

from sklearn.model_selection import train_test_split
# lecture du dataset

df = pd.read_csv('../input/CHERNAIRedited.csv')
# affichage bref du dataframe

df
# exemples des premieres lignes

df.head()
# comptage du nombre de valeurs pour chaque colonne

df.count()
# récupération des labels des colonnes

df.columns
# description en termes de moyenne mini maxi des colonnes numériques

df.describe()
# qq infos

df.info
df['Cs 134 (Bq/m3)'] = pd.to_numeric(df['Cs 134 (Bq/m3)'],errors='coerce')

sns.jointplot("I 131 (Bq/m3)", "Cs 134 (Bq/m3)", df, kind='kde')
df['I 131 (Bq/m3)'] = pd.to_numeric(df['I 131 (Bq/m3)'],errors='coerce')

sns.distplot(df['I 131 (Bq/m3)'], color='blue')
df['Cs 134 (Bq/m3)'] = pd.to_numeric(df['Cs 134 (Bq/m3)'],errors='coerce')

sns.distplot(df['Cs 134 (Bq/m3)'], color='blue')
df['Cs 137 (Bq/m3)'] = pd.to_numeric(df['Cs 137 (Bq/m3)'],errors='coerce')

sns.distplot(df['Cs 137 (Bq/m3)'], color='blue')
print("La longitude mini est de : ")

print(df.X.min())



print("La longitude maxi est de : ")

print(df.X.max())



print("La latitude mini est de : ")

print(df.Y.min())



print("La latitude maxi est de :")

print(df.Y.max())
df.plot(kind="scatter", x="X", y="Y", c="I 131 (Bq/m3)", cmap="rainbow", s=3, figsize=(12,12))

df.groupby(['Date'])['I 131 (Bq/m3)'].mean().plot(kind = 'bar', figsize=(12,8))
df.groupby(['Date'])['Cs 134 (Bq/m3)'].mean().plot(kind = 'bar', figsize=(12,8))
df.groupby(['Date'])['Cs 137 (Bq/m3)'].mean().plot(kind = 'bar', figsize=(12,8))
df1 = df.drop(['PAYS','Ville','Date','End of sampling','Duration(h.min)'], axis=1)

X = df1.drop(['I 131 (Bq/m3)'], axis=1)

y = df1['I 131 (Bq/m3)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
from sklearn import ensemble

rf = ensemble.RandomForestRegressor()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)

print(rf.score(X_test,y_test))
import folium

from folium.plugins import MarkerCluster
#Define coordinates of where we want to center our map

boulder_coords = [47.583042, 12.70]



#Create the map

cher_map = folium.Map(location = boulder_coords, zoom_start = 5)



# Make an empty map

m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)



# I can add marker one by one on the cher_map

for i in range(0,len(df)):

    folium.Marker([df.iloc[i]['Y'], df.iloc[i]['X']], popup=df.iloc[i]['Ville']).add_to(cher_map)



#Display the map

cher_map