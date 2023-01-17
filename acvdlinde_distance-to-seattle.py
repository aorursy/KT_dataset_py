# Thanks to the stackoverflow community for inspiration and pieces of code.

# This kernel calculates the  distance of each  Starbucks to the city of Seattle,

# and determines median distance in kilometers.

import pandas as pd

import numpy as np

from math import radians, cos, sin, asin, sqrt
# create DataFrame 'df' from the Kaggle hosted file



df = pd.read_csv('../input/directory.csv')
# rename the original Longitude, Latitude columns for ease of reference



df = df.rename(columns={'Longitude': 'lon1', 'Latitude': 'lat1'})
# great circle distance formula, vectorized



def haversine_np(lon1, lat1, lon2, lat2):

   

   

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
# decimal GPS coordinates of Seattle. 



lon2, lat2 = -122.335167, 47.6062
# add new column "Distance" to the dataframe, applying the formula from above.

# note how lon2 and lat2 are hardwired (see [6]) and lon1 and lat2 are sourced from the dataframe DF



df['Distance'] = np.vectorize(haversine_np)(df['lon1'], df['lat1'], lon2, lat2)
# show some statistics on the calculated Distances (from Seattle)



df['Distance'].describe()
# next version: try to calculate distances between every Starbucks (i.e. 25599! combinations)

# by replacing Seattle with all 25599 minus one locations