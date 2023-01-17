import pandas as pd

import folium



data = pd.read_csv('../input/train.csv')



subset = data[['pickup_latitude','pickup_longitude']].loc[::10]

average_pickup_location = [data['pickup_latitude'].mean(),data['pickup_longitude'].mean()]

base_map = folium.Map(location=average_pickup_location,tiles = 'OpenStreetMap')

fg = folium.FeatureGroup(name = 'My map')



for index in range(len(subset)):

    location = list(subset.loc[index].values)

    fg.add_child( folium.CircleMarker( location=location,radius=1) )

base_map.add_child(fg)

base_map.save('NYtaxi_pickup.html')