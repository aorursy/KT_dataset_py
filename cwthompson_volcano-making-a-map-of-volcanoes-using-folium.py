import numpy as np

import pandas as pd

import folium

import matplotlib.pyplot as plt
DATA_DIR = '../input/volcano-eruptions/'
data_eruptions = pd.read_csv(DATA_DIR + 'eruptions.csv')

data_events = pd.read_csv(DATA_DIR + 'events.csv')

data_sulfur = pd.read_csv(DATA_DIR + 'sulfur.csv')

data_tree_rings = pd.read_csv(DATA_DIR + 'tree_rings.csv')

data_volcano = pd.read_csv(DATA_DIR + 'volcano.csv')
data_volcano.head(5)
print('Rows:   ', data_volcano.shape[0])

print('Columns:', data_volcano.columns.values)
volcano_map = folium.Map()



# Add each volcano to the map

for i in range(0, data_volcano.shape[0]):

    volcano = data_volcano.iloc[i]

    folium.Marker([volcano['latitude'], volcano['longitude']], popup=volcano['volcano_name']).add_to(volcano_map)



volcano_map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))



volcano_country = pd.DataFrame(data_volcano.groupby(['country']).size()).sort_values(0, ascending=True)

volcano_country.columns = ['Count']

volcano_country.tail(10).plot(kind='barh', legend=False, ax=ax1)

ax1.set_title('Number of Volcanoes per Country')

ax1.set_ylabel('')

ax1.set_xlabel('count')



volcano_region = pd.DataFrame(data_volcano.groupby(['region']).size()).sort_values(0, ascending=True)

volcano_region.columns = ['Count']

volcano_region.tail(10).plot(kind='barh', legend=False, ax=ax2)

ax2.set_title('Number of Volcanoes per Region')

ax2.set_ylabel('')

ax2.set_xlabel('count')



plt.tight_layout()

plt.show()
data_eruptions.columns
data_eruptions[data_eruptions['vei'] > 6]
# Get max VEI for each volcano

volcano_max_vei = data_eruptions.groupby(['volcano_number'])['vei'].max().reset_index()



# Merge these values into the volcano dataframe

data_volcano = pd.merge(data_volcano, volcano_max_vei, on='volcano_number')
def vei_radius(vei):

    return 2 ** (int(vei) - 4) + 3 if not np.isnan(vei) else 1

    

volcano_with_vei = data_volcano#.dropna(subset=['vei'])



# Create the map

volcano_vei_map = folium.Map()



# Create layers

layers = []

for i in range(8):

    layers.append(folium.FeatureGroup(name='VEI: '+str(i)))

layers.append(folium.FeatureGroup(name='VEI: NaN'))



# Add each volcano to the correct layer

for i in range(0, volcano_with_vei.shape[0]):

    volcano = volcano_with_vei.iloc[i]

    # Create marker

    marker = folium.CircleMarker([volcano['latitude'],

                                  volcano['longitude']],

                                  popup=volcano['volcano_name'] + ', VEI: ' + str(volcano['vei']),

                                  radius=vei_radius(volcano['vei']),

                                  color='red' if not np.isnan(volcano['vei']) and int(volcano['vei']) == 7 else 'blue',

                                  fill=True)

    # Add to correct layer

    if np.isnan(volcano['vei']):

        marker.add_to(layers[8])

    else:

        marker.add_to(layers[int(volcano['vei'])])



# Add layers to map

for layer in layers:

    layer.add_to(volcano_vei_map)

folium.LayerControl().add_to(volcano_vei_map)



volcano_vei_map
tectonic_plates = pd.read_csv('../input/tectonic-plate-boundaries/all.csv')

tectonic_plates.head()
plate_map = folium.Map()



plates = list(tectonic_plates['plate'].unique())

for plate in plates:

    plate_vals = tectonic_plates[tectonic_plates['plate'] == plate]

    lats = plate_vals['lat'].values

    lons = plate_vals['lon'].values

    points = list(zip(lats, lons))

    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]

    for i in range(len(indexes) - 1):

        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, color='green', fill=False).add_to(plate_map)



plate_map
def vei_radius(vei):

    return 2 ** (int(vei) - 4) + 3 if not np.isnan(vei) else 1

    

volcano_with_vei = data_volcano#.dropna(subset=['vei'])



# Create the map

complete_map = folium.Map()



# Add tectonic plates to map

plate_layer = folium.FeatureGroup(name='Tectonic Plates')

plates = list(tectonic_plates['plate'].unique())

for plate in plates:

    plate_vals = tectonic_plates[tectonic_plates['plate'] == plate]

    lats = plate_vals['lat'].values

    lons = plate_vals['lon'].values

    points = list(zip(lats, lons))

    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]

    for i in range(len(indexes) - 1):

        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, color='green', fill=False).add_to(plate_layer)

plate_layer.add_to(complete_map)



# Create layers

layers = []

for i in range(8):

    layers.append(folium.FeatureGroup(name='VEI: '+str(i)))

layers.append(folium.FeatureGroup(name='VEI: NaN'))



# Add each volcano to the correct layer

for i in range(0, volcano_with_vei.shape[0]):

    volcano = volcano_with_vei.iloc[i]

    # Create marker

    marker = folium.CircleMarker([volcano['latitude'],

                                  volcano['longitude']],

                                  popup=volcano['volcano_name'] + ', VEI: ' + str(volcano['vei']),

                                  radius=vei_radius(volcano['vei']),

                                  color='red' if not np.isnan(volcano['vei']) and int(volcano['vei']) == 7 else 'blue',

                                  fill=True)

    # Add to correct layer

    if np.isnan(volcano['vei']):

        marker.add_to(layers[8])

    else:

        marker.add_to(layers[int(volcano['vei'])])



# Add layers to map

for layer in layers:

    layer.add_to(complete_map)



# Add layer control

folium.LayerControl().add_to(complete_map)



complete_map