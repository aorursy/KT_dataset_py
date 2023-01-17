import numpy as np 
import pandas as pd 
import folium
from folium.plugins import HeatMap
import seaborn as sns
from functools import reduce
import geopandas as gpd
kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva_loans.activity.describe()
activity_category_sums = kiva_loans.activity.value_counts()
activity_category_sums.loc[activity_category_sums > 4000]
# May need to excluse Fish Selling beacuse most of these have resale, which may not be indicative of low level of relative poverty
kiva_loans.loc[(kiva_loans.activity != 'Fish Selling')] 
satellite_countries = ('Moldova', 'Estonia', 'Latvia', 'Lithuania', 'Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Russia', 'Armenia', 'Azerbaijan', 'Georgia', 'Ukraine')
#print(satellite_countries)
#
kiva_loans_satellite = kiva_loans.loc[kiva_loans.country.isin(satellite_countries)]
kiva_loans_satellite.country.value_counts()
kiva_loans_satellite.activity.value_counts()
kiva_loans_satellite.loc[kiva_loans_satellite.country == 'Kyrgyzstan']
def create_heatmap_coords(row, metric, coord1, coord2):
    return [row[coord1], row[coord2], row[metric]]

def create_coords(row, coord1, coord2):
    return [row[coord1], row[coord2]]

def create_heatmap_coords_column(df, col_title='heatmap_coords', metric='MPI', coord1='longitude', coord2='latitude'):
    df[col_title] = df.apply(lambda x: create_heatmap_coords(x, metric, coord1, coord2), axis=1)
    
def create_coords_column(df, coord1='longitude', coord2='latitude'):
    df['coords'] = df.apply(lambda x: create_coords(x, coord1, coord2), axis=1)
    
def get_coords_of_n_smallest(df, name='Country', metric='MPI', n=10):
    return np.array(df.sort_values(by=[metric])[['coords', name, metric]][:n])

def get_coords_of_n_largest(df, name='Country', metric='MPI', n=10):
    return np.array(df.sort_values(by=[metric])[['coords', name, metric]][-n:])

kiva_mpi_region_locations = kiva_mpi_region_locations.dropna()
kiva_mpi_region_locations = kiva_mpi_region_locations[(kiva_mpi_region_locations['LocationName'] != 'Lac, Chad') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Logone Occidental, Chad') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Logone Oriental, Chad') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Kanem, Chad') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Hama, Syrian Arab Republic') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Tortous, Syrian Arab Republic') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Gharbia, Egypt') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Matroh, Egypt') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Port Said, Egypt') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Bogota, Colombia') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Orinoquia Y Amazonia, Colombia') & 
                                                      (kiva_mpi_region_locations['LocationName'] != 'Central-Eastern, Uzbekistan') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Southern, Uzbekistan') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'Eastern, Uzbekistan') &
                                                      (kiva_mpi_region_locations['LocationName'] != 'St. Ann, Jamaica') 
                                                     ]
create_heatmap_coords_column(kiva_mpi_region_locations, coord1='lat', coord2='lon')
create_coords_column(kiva_mpi_region_locations, coord1='lat', coord2='lon')
lowest_MPI = get_coords_of_n_smallest(kiva_mpi_region_locations, name='LocationName')
highest_MPI = get_coords_of_n_largest(kiva_mpi_region_locations, name='LocationName')
def add_markers_to_map(head_map, coords, metric='MPI', num_markers=10, color='green', icon='info-sign', prefix='glyphicon'):
    for i in range(num_markers):
        folium.Marker(
            location=coords[i][0],
            icon=folium.Icon(color=color, icon=icon, prefix=prefix),
            popup='{}, {} {}'.format(coords[i][1], metric, coords[i][2])
        ).add_to(head_map)

common_map = folium.Map(location=[10, 0], zoom_start=3)
hm = HeatMap(kiva_mpi_region_locations['heatmap_coords'], radius=15, blur=5)
hm.add_to(common_map)
add_markers_to_map(common_map, lowest_MPI)
add_markers_to_map(common_map, highest_MPI, color='red')
common_map
country_mappings_a = {
    'United Kingdom': 'UK',
    'United States': 'US',
    'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.': 'Yemen',
    'West Bank and Gaza': 'Palestine',
    'Korea, Rep.': 'South Korea',
    'Korea, Dem. People’s Rep.': 'North Korea',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'St. Martin (French part)': 'St. Martin',
    'Syrian Arab Republic': 'Syria',
    'Micronesia, Fed. Sts.': 'Micronesia',
    'Russian Federation': 'Russia',
    'Macedonia, FYR': 'Macedonia',
    'Macao SAR, China': 'Macau',
    'Iran, Islamic Rep.': 'Iran',
    'Hong Kong SAR, China': 'Hong Kong',
    'Egypt, Arab Rep.': 'Egypt',
    'Virgin Islands (U.S.)': 'U.S. Virgin Islands',
    'Congo, Dem. Rep.': 'Congo - Kinshasa',
    'Congo, Rep.': 'Congo - Brazzaville',
    'Brunei Darussalam': 'Brunei',
    'Bahamas, The': 'Bahamas',
    'Gambia, The': 'Gambia'
}

country_mappings_b = {
    'Macedonia, The former Yugoslav Republic of': 'Macedonia',
    'Moldova, Republic of': 'Moldova',
    'Syrian Arab Republic': 'Syria',
    'Viet Nam': 'Vietnam',
    "Lao People's Democratic Republic": 'Laos',
    'Central African Republic': 'Central African Rep.',
    'Congo, Democratic Republic of the': 'Dem. Rep. Congo',
    'Congo, Republic of': 'Congo',
    "Cote d'Ivoire": "CÃ´te d'Ivoire",
    'Tanzania, United Republic of': 'Tanzania'
}
