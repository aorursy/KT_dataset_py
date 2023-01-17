# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



!pip install scikit-bio

import skbio

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read csv

world_location_dataset = pd.read_csv("../input/world-cities-datasets/worldcities.csv")

world_location_dataset
#Get US state

us_location_dataset = world_location_dataset[world_location_dataset['iso2']=="US"]

us_location_dataset
#Get top 15 cities based on index of US state

selected_cities_location = us_location_dataset.head(15)

selected_cities_location
#for each cities, get the latitude and longitude only

selected_cities_location = selected_cities_location[["city","lat","lng"]]

selected_cities_location = selected_cities_location.reset_index()

selected_cities_location
#Calculate distance from/to each cities (Euclidean)

def distance(x1,y1,x2,y2):

    return (round(((abs(x2)-abs(x1))**2 + (abs(y2)-abs(y1))**2)**0.5,2))



#Create distance matrix

dmat = np.zeros((15,15))



for i in range(15):

    for j in range(i,15):

        if(i == j):

            pass

        elif(i > j):

            break

        else:

            x1,y1 = selected_cities_location.loc[i,['lat','lng']]

            x2,y2 = selected_cities_location.loc[j,['lat','lng']]

            calc = distance(x1,y1,x2,y2)

            dmat[i][j] = calc

            dmat[j][i] = calc
#Visualization Distance Matrix

dmat_df = pd.DataFrame(dmat)

dmat_df
#Visualization

fig, ax = plt.subplots()

ax.scatter(selected_cities_location['lat'],selected_cities_location['lng'])



for i, txt in enumerate(selected_cities_location['city']):

    ax.annotate(txt, (selected_cities_location['lat'][i],selected_cities_location['lng'][i]))



ax.title.set_text("15 US Cities")

ax.set_xlabel("Latitude")

ax.set_ylabel("Longitude")



#Ref: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
PCoA = skbio.stats.ordination.pcoa(dmat)

PCoA
#fig = PCoA.plot(df=selected_cities_location, column="city", title='15 US Cities', cmap='Set1', s=50)

fig = PCoA.plot(title='15 US Cities', cmap='Set1', s=50)