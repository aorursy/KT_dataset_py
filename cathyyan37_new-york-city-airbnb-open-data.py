import pandas as pd

import numpy as np
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



# Tableau needs a "hierarchy" of locations to map data correctly

data['country'] = 'United States'

data['state'] = 'New York'

data['city'] = 'New York City'



# Select only relevant columns

data = data[['country', 'state', 'city', 'neighbourhood_group', 'latitude', 'longitude', 'room_type', 'price']]

print(data.head())
data.to_csv('airbnb_nyc',index=False)