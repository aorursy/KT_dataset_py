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
ratingsdf = pd.read_csv('../input/ml-challenge/ratings.csv')

ratingsdf.count()
ratingsdf['venue_id'].nunique()
uniqueVenuedf = ratingsdf['venue_id']

uniqueVenuedf.drop_duplicates(inplace=True)

uniqueVenuedf.count()
uniqueVenuedf.head()
venuesdf = pd.read_csv('../input/data-exploration/correct_venues.csv')

venuesdf.count()

result = pd.merge(venuesdf,uniqueVenuedf, left_on='id', right_on='venue_id')

result = result.drop('venue_id', axis=1)

result.head(20)
result.shape
top10df = pd.read_csv('../input/data-exploration/top10checkins.csv')

top10df.head()
result['tmp'] = 1

top10df['tmp'] = 1

combinedf = pd.merge(top10df, result, on=['tmp'])

combinedf = combinedf.drop('tmp', axis=1)

combinedf.head()
combinedf.shape
combinedf.to_csv('combine_user_venues.csv', index=False)