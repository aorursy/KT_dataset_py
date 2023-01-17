import numpy as np 

import pandas as pd

pd.set_option('display.max_columns', 38)



from geopy.geocoders import Nominatim

from datetime import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/caraccidentnypd/qiz3-axqb.csv')

df.info()
df.date = df.date.astype('datetime64')

df.head(10)
def get_zipcode_via_lat_long(latitude, longitude):

    location = geolocator.reverse((latitude, longitude))

    zipcode = int(location.raw['address']['postcode'])

    return zipcode
def get_zipcode_via_address(address):

    location = geolocator.geocode(address)

    s = pd.DataFrame(location.raw).loc[0,'display_name'] #location.raw can have duplicates, so, we use 0th row 

    zipcode = int(s.split(sep=', ')[-2])

    return zipcode
def get_accidents_within_time_interval(initial_date, final_date, zipcode):

    

    if zipcode in df.zip_code.unique():

            

        if type(initial_date) is str:

            initial_date = datetime.strptime(initial_date, '%Y-%m-%d')

            final_date = datetime.strptime(final_date, '%Y-%m-%d')

            

        assert initial_date < final_date, 'Initial date was given as later than final date.'

        

        df2 = df[df.zip_code == zipcode].sort_values(by='date')

        result = df2[(df2.date>=initial_date) & (df2.date<=final_date)]

        

        if len(result) == 0:

            print('No accidents in zipcode {} within the given time interval.'.format(zipcode))

        return result

    

    else:

        raise Exception('Zipcode {} does not exist.'.format(zipcode))
def get_accident_rate(zipcode = None, address = None):

    if zipcode is not None:

        

        if zipcode in df.zip_code.unique():

            return df.zip_code.count_values()[zipcode]/len(df.zip_code)

        else:

            raise Exception('Zipcode {} does not exist.'.format(zipcode))

    

    #if address not None:

        #dataset has no address
lat = df.loc[1,'latitude']

long = df.loc[1, 'longitude'] 

geolocator = Nominatim(user_agent="fc")

get_zipcode_via_lat_long(lat, long)
i_date = datetime(2019,6,29)

f_date = datetime(2019,7,30)



get_accidents_within_time_interval(i_date, f_date, 10022)
location = geolocator.reverse((lat, long))

location.raw
accidents = pd.read_csv('../input/usaccidents20161q/US_Accidents_2016.csv', sep=';')

accidents.info()
accidents.head(50)
accidents.Start_Time = accidents.Start_Time.astype(datetime)

accidents.End_Time = accidents.End_Time.astype(datetime)

accidents.Weather_Timestamp = accidents.Weather_Timestamp.astype(datetime)

accidents.head()
','.join(accidents[:,])