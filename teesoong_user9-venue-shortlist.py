# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from geopy import distance

def calculate_distance(lat1, long1, lat2, long2):

    user_loc = (lat1, long1)

    venue_loc = (lat2, long2)

    return distance.distance(user_loc, venue_loc).km
combinedf = pd.read_csv('../input/prediction-preparation/combine_user_venues.csv')

partialdf = combinedf[combinedf['user_id'] == 439413]

partialdf.shape
partialdf.nunique()
ratingsdf = pd.read_csv('../input/data-exploration/avgratings.csv')

ratedvenuesdf = ratingsdf[ratingsdf['user_id'] == 439413]

ratedvenuesdf.head()
cond = partialdf['id'].isin(ratedvenuesdf['venue_id'])

partialdf.drop(partialdf[cond].index, inplace = True)

partialdf.shape
partialdf['distance'] = partialdf.apply(lambda row : calculate_distance(row['user_latitude'], row['user_longitude'], row['latitude'], row['longitude']), axis=1)

partialdf.head()
shortlistdf = partialdf[partialdf['distance'] < 80]

shortlistdf.head()
shortlistdf.shape
shortlistdf.to_csv('shortlist_user9.csv', index=False)