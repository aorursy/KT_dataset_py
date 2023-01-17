# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from ast import literal_eval



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/registered-business-locations-san-francisco.csv')
data.head()
# Takes in a list of datetimes and returns a list of years

def get_business_year(arr):

    years = []

    for d in arr:

        if str(d) == 'nan':

            years.append("NaN")

        else:

            years.append(datetime.strptime(str(d).split('T')[0], '%Y-%m-%d').year)

    

    return years
years = get_business_year(data['Business Start Date'])

data['Business Year'] = years

data[['Business Start Date', 'Business Year']]
def get_dict(my_dict, val, default='NaN'):

    if val in my_dict:

        return my_dict[val]

    else:

        return default
# Takes a list of dictionary strings with geographic data and returns the latitude and longitude in separate lists

def get_geo_data(arr):

    lat = []

    long = []

    for geo_str in arr:

        if str(geo_str) == 'nan':

            lat.append("NaN")

            long.append("NaN")

        else:

            geo_dict = literal_eval(geo_str)

#             print(geo_dict)

            lat.append(get_dict(geo_dict,'latitude'))

            long.append(get_dict(geo_dict,'longitude'))

    return lat, long
lat, long = get_geo_data(data['Business Location'])

data['Latitude'] = lat

data['Longitude'] = long
data[['Business Location', 'Latitude', 'Longitude']]
## There are a number of NaN (null) values. So don't be worried from ouptut above

data.loc[data['Business Location'].isnull()]
data.loc[data['Business Location'].notnull(), ['Business Location', 'Latitude', 'Longitude']]