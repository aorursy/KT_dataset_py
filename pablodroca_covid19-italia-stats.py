import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
data = pd.read_csv('/kaggle/input/covid19-italia-simulazione-di-casi-e-deceduti/italia_nuovi_casi.csv')
cases_data = data[data['tipo'] == 'positivi']
cases_data.drop(columns=['tipo'], inplace=True)
deaths_data = data[data['tipo'] == 'deceduti']
deaths_data.drop(columns=['tipo'], inplace=True)
regions_data = pd.read_csv('/kaggle/input/covid19-italia-simulazione-di-casi-e-deceduti/italia_regioni.csv')
print(cases_data)
print(regions_data)
regions_values = regions_data.copy().to_numpy()
def map_closest_region(row):
    lat1 = row.lat
    lon1 = row.long
    closest_region = np.nan
    closest_distance = None
    for reg_row in regions_values:
        lon2 = reg_row[2]
        lat2 = reg_row[1]
        distance = haversine_np(lon1, lat1, lon2, lat2)
        if not closest_distance or closest_distance > distance:
            closest_region = reg_row[0]
            closest_distance = distance
    return closest_region
cases_data['regione'] = cases_data.apply(map_closest_region, axis=1)
deaths_data['regione'] = deaths_data.apply(map_closest_region, axis=1)
positivi_totali = cases_data.groupby(['data']).count()['regione'].to_frame(name='totale_positivi')
deceduti_totali = deaths_data.groupby(['data']).count()['regione'].to_frame(name='totale_deceduti')
totali = pd.concat([positivi_totali, deceduti_totali])
print(totali)
print('---')
regione_top_3 = cases_data.groupby(['regione']).count()['data'].to_frame(name='totale_positivi').sort_values('totale_positivi', ascending=False).head(3)
print(regione_top_3)
print('---')
porcentuale_deceduti = (deaths_data.count()['data'] / cases_data.count()['data']) * 100
print('Porcentuale Deceduti={:.2f}%'.format(porcentuale_deceduti))
