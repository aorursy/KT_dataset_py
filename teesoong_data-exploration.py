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
df = pd.read_csv('../input/ml-challenge/checkins.csv')

df.count()
df.head(20)
df.tail(20)
df.isna().sum()
df.isin([0]).sum()
df.nunique()
duplicateRowsDF = df[df.duplicated(['user_id', 'venue_id','created_at'])]



duplicateRowsDF.count()
noDupdf = df.drop_duplicates(subset=['user_id', 'venue_id','created_at'], inplace=False)

noDupdf.count()
noDupdf['user_id'].value_counts().head(15)
top10df = noDupdf['user_id'].value_counts().head(14).rename_axis('user_id').reset_index(name='checkin_counts')
userdf = pd.read_csv('../input/ml-challenge/users.csv')

userdf.head()
mergeuserdf = pd.merge(left=top10df, right=userdf, left_on='user_id', right_on='id')

mergeuserdf
import folium

from folium import Choropleth, Circle, Marker

lat = mergeuserdf.iloc[0]['latitude']

long = mergeuserdf.iloc[0]['longitude']

usermap = folium.Map(location=[lat, long], tiles='cartodbpositron', zoom_start=3)

# Add a bubble map to the base map

for i in range(0,len(mergeuserdf)):

      Circle(

        location=[mergeuserdf.iloc[i]['latitude'], mergeuserdf.iloc[i]['longitude']],

        radius=20,

        color='darkred').add_to(usermap)

usermap
mergeuserdf = mergeuserdf.drop('id', axis=1)

mergeuserdf.rename(columns = {'latitude':'user_latitude', 'longitude':'user_longitude'}, inplace = True)

mergeuserdf.to_csv('top10checkins.csv', index=False)
ratingsdf = pd.read_csv('../input/ml-challenge/ratings.csv')

pd.options.display.float_format = "{:.2f}".format

ratingsdf.describe()
dupRatingsDF = ratingsdf[ratingsdf.duplicated(['user_id', 'venue_id'])]



dupRatingsDF.count()
avgratingsdf = ratingsdf.groupby(['user_id','venue_id']).mean().reset_index()

avgratingsdf.head(10)
avgratingsdf.shape
avgratingsdf.to_csv('avgratings.csv', index=False)
venuesdf = pd.read_csv('../input/ml-challenge/venues.csv')

venuesdf.describe()
print(venuesdf[venuesdf.latitude > 90])
venuesdf = venuesdf[venuesdf['latitude'] <= 90]

venuesdf.describe()
venuesdf.to_csv('correct_venues.csv', index=False)