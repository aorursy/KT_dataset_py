
!pip install python-googlegeocoder
PAGE = 0 # can be between 0 and 9



import pandas as pd
from tqdm import tqdm
df = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

locations = df.drop_duplicates('location').iloc[PAGE::10] # showing every 10th location - since there are way too many!

import folium

locs = folium.Map(location=[45, -122],zoom_start=5)

from googlegeocoder import GoogleGeocoder
KEY = '???????????????????????????????????????' # sry, had to remove this, can be obtained here -- https://console.cloud.google.com/google/maps-apis/credentials

def get_latlon(location):
    geocoder = GoogleGeocoder(KEY)
    search = geocoder.get(location)
    return (search[0].geometry.location.lat, search[0].geometry.location.lng)

for d in tqdm(locations.to_dict(orient='records')):
    
    lat, lon = d['latitude'],d['longitude']
    
    if lat == 'Not specified':
        try:
            lat, lon=get_latlon(d['location'])
        except KeyboardInterrupt:
            raise
        except:
            continue
    
    folium.Marker((lat, lon), popup=d['location']).add_to(locs)
        
locs
