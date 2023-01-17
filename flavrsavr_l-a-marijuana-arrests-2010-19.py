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



# save filepath to variable for easier access

weedbusts_file_path = '../input/Marijuana_Data.csv'

# read the data and store data in DataFrame titled melbourne_data

weedbusts_data = pd.read_csv(weedbusts_file_path) 
weedbust_features = ['Arrest Date', 'Time', 'Area ID', 'Area Name',

       'Reporting District', 'Age', 'Sex Code', 'Descent Code',

       'Charge Group Code', 'Arrest Type Code',

       'Charge', 'Charge Description', 'Location']

# Descent Code is race; Reporting District is numerical; location is a lat/long pair with ()s and comma.



X = weedbusts_data[weedbust_features]
lat_long = X['Location'].str.strip('()').str.split(',', expand=True) # returns Series, not affecting actual df



X = X.assign(Latitude = lat_long[0], Longitude = lat_long[1])



# X.describe

X = X.assign(ArrestDateObj = pd.to_datetime(X['Arrest Date']))



X.columns

# stick with just year, for simplicity

grouped = X.groupby([X["ArrestDateObj"].dt.year]).count()

grouped['ArrestDateObj'].plot(kind='bar') # use count of arrests col

# https://stackoverflow.com/questions/27365467/can-pandas-plot-a-histogram-of-dates#29036738

grouped = X.groupby([X["ArrestDateObj"].dt.year, X["ArrestDateObj"].dt.month])['ArrestDateObj'] # .count()

grouped_count = grouped.count()

grouped_count.plot(kind='bar', figsize=(16,6))
# try scatterplotting arrests by lat, long



# reduce data set first, maybe use 2018 data

subset2018 = X.loc[X["ArrestDateObj"].dt.year == 2018]

locs18 = subset2018[['Latitude','Longitude']]

locs18
# try plotting subset2018 by lat, long



# need to coerce lat, long to floats (orig were strings)

# type(locs18.iloc[0,1]) -- str

locs18['Latitude'] = locs18['Latitude'].astype('float64')

locs18['Longitude'] = locs18['Longitude'].astype('float64')

locs18.dtypes
locs18.plot('Latitude','Longitude', kind='scatter', figsize=(16,12))

# kinda doesn't look like anything without a map!

subset_year = 2010

subset = X.loc[X["ArrestDateObj"].dt.year == subset_year]

locs_subset = subset[['Latitude','Longitude']]

locs_subset['Latitude'] = locs_subset['Latitude'].astype('float64')

locs_subset['Longitude'] = locs_subset['Longitude'].astype('float64')

locs_subset.plot('Latitude','Longitude', kind='scatter')
# 'keep only' action example: df[np.abs(df.Data-df.Data.mean()) <= (3*df.Data.std())]

# NOTE: This doesn't remove data permanently, only in this 1-liner plot command.

locs_subset[(locs_subset.Latitude != 0) & (locs_subset.Longitude < -118.0)].plot('Latitude','Longitude', kind='scatter', figsize=(16,12))

import geopandas as gp

import folium



avg_lat = locs_subset.Latitude.mean()

avg_lon = locs_subset.Longitude.mean()



m = folium.Map(location=[avg_lat,avg_lon])

# add a marker for every record in the filtered data, use a clustered view

for index, bust in locs_subset[:].iterrows():  # interrows outputs [index,Series] naturally, so split on for statement to avoid having to [1] index later

    folium.Circle(

        location = [bust['Latitude'], bust['Longitude']], 

        radius=3, # units: meters

        color='crimson',

        popup=None,

        fill=True,

        fill_color='crimson',

    ).add_to(m)

  

m