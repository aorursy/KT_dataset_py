% matplotlib inline

###############################################
# Import Miscellaneous Assets
###############################################
import numpy as np
import pandas as pd
import json
from datetime import datetime
from dateutil.parser import parse as date_parse
from collections import Counter
from pprint import pprint as pp
from tqdm import tqdm, tqdm_notebook
from IPython.display import display
import warnings
import os
import sys
from geopy.distance import vincenty, great_circle

###############################################
# Import Plotting Assets
###############################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.pyplot import subplot, figure
import seaborn as sns
import folium
from folium import plugins as folium_plugins
from folium import features

###############################################
# Declare Global Variables
###############################################
plt.interactive(False)
sns.set_style('whitegrid')
pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter(action='ignore', category=FutureWarning)

weather_set_dir = '../input/rrv-weather-data'
weather_data_dir = '{}/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather'.format(weather_set_dir)
original_data_dir = '../input/recruit-restaurant-visitor-forecasting'

weather_columns = [
    'avg_temperature', 'high_temperature', 'low_temperature', 'precipitation',
    'hours_sunlight', 'solar_radiation', 'deepest_snowfall', 'total_snowfall', 'avg_wind_speed',
    'avg_vapor_pressure', 'avg_local_pressure', 'avg_humidity', 'avg_sea_pressure', 
    'cloud_cover'
]
air_store_info = pd.read_csv('{}/air_store_info_with_nearest_active_station.csv'.format(weather_set_dir))
hpg_store_info = pd.read_csv('{}/hpg_store_info_with_nearest_active_station.csv'.format(weather_set_dir))

air_station_distances = pd.read_csv('{}/air_station_distances.csv'.format(weather_set_dir))
hpg_station_distances = pd.read_csv('{}/hpg_station_distances.csv'.format(weather_set_dir))

weather_stations = pd.read_csv('{}/weather_stations.csv'.format(weather_set_dir))
nearby_active_stations = pd.read_csv('{}/nearby_active_stations.csv'.format(weather_set_dir))
feature_manifest = pd.read_csv('{}/feature_manifest.csv'.format(weather_set_dir))
original_as_info = pd.read_csv('{}/air_store_info.csv'.format(original_data_dir))
original_hs_info = pd.read_csv('{}/hpg_store_info.csv'.format(original_data_dir))

display(original_as_info.head(5))
display(original_hs_info.head(5))

print('Air Equal: {}'.format(original_as_info.equals(air_store_info[original_as_info.columns])))
print('HPG Equal: {}'.format(original_hs_info.equals(hpg_store_info[original_hs_info.columns])))
display(hpg_store_info.head())
print('These columns are added:')
pp([_ for _ in hpg_store_info.columns if _ not in original_hs_info.columns])
print(hpg_station_distances.shape)
display(hpg_station_distances.head())
store = hpg_store_info.iloc[0]
lat_str, lon_str = store['latitude_str'], store['longitude_str']
lookup_coords = '({}, {})'.format(lat_str, lon_str).replace('"', '')
print(lookup_coords)

distances = hpg_station_distances[lookup_coords]
print(distances.values[:5])
closest_station_distance = distances.min()
print('Distance to Closest Station: {} km'.format(closest_station_distance))

ids = hpg_station_distances['station_id'].values
closest_station_id = ids[distances.tolist().index(closest_station_distance)]
print('Closest Station ID: {}'.format(closest_station_id))
stations_in_range = [(ids[_], distances[_]) for _ in range(len(distances)) if distances[_] <= 17]
stations_in_range = sorted(stations_in_range, key=lambda _: _[1], reverse=False)
pp(stations_in_range)

def select_stations(latitude_str, longitude_str, distance_df, effective_range=17.0, date_floor=None, top_n=None):
    """
    Filters stations based on proximity to coordinates, and termination status
    Note: if longitude_str is None, the first argument is assumed to be a properly formatted coordinate string
    :param latitude_str: latitude_str from air/hpg_store_info_with_nearest_active_station
    :param longitude_str: longitude_str from air/hpg_store_info_with_nearest_active_station
    :param distance_df: one of the following DFs: air_station_distances, hpg_station_distances
    :param effective_range: float in kilometers specifying the max distance a station can be from the store
    :param date_floor: if datetime, remove stations terminated before date_floor. If None, ignore termination
    :param top_n: if int, return at most top_n many stations. If None, all stations will be returned
    :returns: a list of tuples of (station_id, distance) that meet the given specifications - sorted by distance
    """
    if longitude_str is not None:
        _lookup_coords = '({}, {})'.format(latitude_str, longitude_str).replace('"', '')
    else:
        _lookup_coords = latitude_str
        
    _ids, _distances = distance_df['station_id'].values, distance_df[_lookup_coords]
    _result = [(_ids[_], _distances[_]) for _ in range(len(_ids)) if _distances[_] <= effective_range]
    
    if date_floor is not None and isinstance(date_floor, datetime):
        _result = [_ for _ in _result if '____' not in _[0] or date_parse(_[0].split('____')[1]) > date_floor]

    return sorted(_result, key=lambda _: _[1])[:top_n]

_test_0 = select_stations(lat_str, lon_str, hpg_station_distances)
_test_1 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('2017-5-31'))
_test_2 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('2017-5-31'), top_n=2)
_test_3 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('1975-12-13'))

assert(_test_0 == stations_in_range)
assert(_test_1 == stations_in_range[:4] + [stations_in_range[-1]])
assert(_test_2 == stations_in_range[:2])
assert(_test_3 == stations_in_range)
print('feature_manifest.shape: {}'.format(feature_manifest.shape))
active_feature_manifest = feature_manifest.loc[~feature_manifest['id'].str.contains('____')]
print('active_feature_manifest.shape: {}'.format(active_feature_manifest.shape))
display(active_feature_manifest.sample(10))
ax = sns.clustermap(active_feature_manifest.drop(['id'], axis=1), figsize=(8, 8))
plt.show()
all_ranges, d_floor = ['10', '17', '20'], date_parse('2017-5-31')
separate_results, combined_results = {_: [] for _ in all_ranges}, {_: [] for _ in all_ranges}

for sys_distances in [air_station_distances, hpg_station_distances]:
    for coords in tqdm(sys_distances.columns.values[3:], leave=False):
        for e_range in all_ranges:
            res = select_stations(coords, None, sys_distances, effective_range=int(e_range), date_floor=d_floor)
            
            separate_results[e_range].extend([_[0] for _ in res if _[0] not in separate_results[e_range]])
            combined_results[e_range].append([_[0] for _ in res])

print('#' * 30 + ' separate_results ' + '#' * 30)
for _r in all_ranges:
    print('{}     {}'.format(_r, len(separate_results[str(_r)])))
print('#' * 30 + ' combined_results ' + '#' * 30)
for _r in all_ranges:
    print('{}     {}'.format(_r, len(combined_results[str(_r)])))
def build_coordinate_count_map(store_info, station_distances):
    info_by_coordinates = {_: 0 for _ in station_distances.columns.values[3:]}

    for i, row in store_info.iterrows():
        coordinate_str = '({}, {})'.format(row['latitude_str'], row['longitude_str']).replace('"', '')
        info_by_coordinates[coordinate_str] += 1

    return pd.DataFrame(
        columns=['coordinates', 'coordinate_count'],
        data=[[_k, info_by_coordinates[_k]] for _k in station_distances.columns.values[3:]]
    )

def filter_coverage(manifest, target_vals, do_isin=False, target_col='id', reindex=True, drop_cols=['id']):
    if do_isin is False:
        _res = pd.DataFrame(
            columns=manifest.columns.values,
            data=[manifest.loc[manifest[target_col] == _, :].values[0] for _ in target_vals]
        )
    else:
        _res = manifest.loc[manifest[target_col].isin(target_vals)]
    
    _res = _res.reset_index(drop=True) if reindex is True else _res
    _res = _res.drop(drop_cols, axis=1) if drop_cols is not None else _res
        
    return _res

air_coord_counts = build_coordinate_count_map(air_store_info, air_station_distances)
hpg_coord_counts = build_coordinate_count_map(hpg_store_info, hpg_station_distances)
coordinate_map = pd.concat((air_coord_counts, hpg_coord_counts)).reset_index(drop=True)
all_coords = air_station_distances.columns.values[3:].tolist() + hpg_station_distances.columns.values[3:].tolist()

# station_ids must be a list of length == len(all_coords)
# station_ids must contain a station_id for each coordinate group in all_coords
# station_ids must also be in the same order as all_coords
# i.e. station_ids[i] is the station for the stores at all_coords[i]
def coverage_by_store(station_ids, drop_cols=['id']):
    all_nearby_station_ids = []
    for i, coords in enumerate(all_coords):
        coord_count = coordinate_map.loc[coordinate_map['coordinates'] == coords, 'coordinate_count'].values[0]
        all_nearby_station_ids.extend([station_ids[i]] * coord_count)
    
    return filter_coverage(feature_manifest, all_nearby_station_ids, drop_cols=drop_cols)
###############################################
# Prepare the Data
###############################################
# Coverage for the nearest station for every store coordinate group
nearest_by_coord_group = filter_coverage(feature_manifest, [_[0] for _ in combined_results['20'] if len(_) > 0])
# Coverage for the nearest station for every store (same values as above, but applied to every store)
nearest_by_store = coverage_by_store([_[0] for _ in combined_results['20']], drop_cols=None)

###############################################
# Plot the Data
###############################################
fig = figure(figsize=(12, 9))
gs = gridspec.GridSpec(13, 3)

cbar_ax = subplot(gs[0, :])
ax1, ax2, ax3 = subplot(gs[1:, 0]), subplot(gs[1:, 1]), subplot(gs[1:, 2])

cbar_kwargs = dict(cbar=True, cbar_ax=cbar_ax, cbar_kws=dict(orientation='horizontal'))
sns.heatmap(active_feature_manifest.drop(['id'], axis=1), ax=ax1, yticklabels=False, cbar=False)
sns.heatmap(nearest_by_coord_group, ax=ax2, yticklabels=False, cbar=False)
sns.heatmap(nearest_by_store.drop(['id'], axis=1), ax=ax3, yticklabels=False, **cbar_kwargs)

ax1.set_title('1) All Stations {}'.format(active_feature_manifest.drop(['id'], axis=1).shape))
ax2.set_title('2) Coord Groups {}'.format(nearest_by_coord_group.shape))
ax3.set_title('3) Stores {}'.format(nearest_by_store.drop(['id'], axis=1).shape))

plt.tight_layout()
plt.show()
weather_data = {_: pd.read_csv('{}/{}.csv'.format(weather_data_dir, _)) for _ in separate_results['20']}

def calculate_coverage(df):
    num_rows = float(df.shape[0])
    null_counts = df.isnull().sum()
    return [((num_rows - float(null_counts[_])) / num_rows) for _ in weather_columns]

def build_daily_combined_coverage(stations):
    summed_df = None
    if len(stations) == 0:
        return np.zeros(14).tolist()

    for station in stations:
        station_df = weather_data[station].drop(['calendar_date'], axis=1)
        summed_df = station_df if summed_df is None else summed_df.add(station_df, fill_value=0)

    return calculate_coverage(summed_df)

combined_coverages = {_: [] for _ in all_ranges}

for i, row in coordinate_map.iterrows():
    for e_range in all_ranges:
        _current_coverage = build_daily_combined_coverage(combined_results[e_range][i])
        combined_coverages[e_range].extend([[row['coordinates']] + _current_coverage] * row['coordinate_count'])

combined_coverage_dfs = {_k: pd.DataFrame(
    columns=['coordinates'] + weather_columns, data=_v
) for _k, _v in combined_coverages.items()}
fig = figure(figsize=(12, 15))
gs = gridspec.GridSpec(19, 2)

cbar_ax = subplot(gs[0, :])
ax1, ax2 = subplot(gs[1:10, 0]), subplot(gs[1:10, 1])
ax3, ax4 = subplot(gs[10:, 0]), subplot(gs[10:, 1])

cbar_kwargs = dict(cbar=True, cbar_ax=cbar_ax, cbar_kws=dict(orientation='horizontal'))
sns.heatmap(nearest_by_store.drop(['id'], axis=1), ax=ax1, yticklabels=False, xticklabels=False, cbar=False)
sns.heatmap(combined_coverage_dfs['10'].drop(['coordinates'], axis=1), ax=ax2, yticklabels=False, xticklabels=False, cbar=False)
sns.heatmap(combined_coverage_dfs['17'].drop(['coordinates'], axis=1), ax=ax3, yticklabels=False, cbar=False)
sns.heatmap(combined_coverage_dfs['20'].drop(['coordinates'], axis=1), ax=ax4, yticklabels=False, **cbar_kwargs)

ax1.set_title('1) Where We Left Off - No Combinations {}'.format(nearest_by_store.drop(['id'], axis=1).shape))
ax2.set_title('2) Combined - 10 km {}'.format(combined_coverage_dfs['10'].drop(['coordinates'], axis=1).shape))
ax3.set_title('3) Combined - 17 km {}'.format(combined_coverage_dfs['17'].drop(['coordinates'], axis=1).shape))
ax4.set_title('4) Combined - 20 km {}'.format(combined_coverage_dfs['20'].drop(['coordinates'], axis=1).shape))

plt.tight_layout()
plt.show()
def coord_format(row):
    return '({}, {})'.format(row['latitude_str'], row['longitude_str']).replace('"', '')

air_store_info['coordinates'] = air_store_info.apply(coord_format, axis=1)
hpg_store_info['coordinates'] = hpg_store_info.apply(coord_format, axis=1)

def clean_string(a_string):
    if sys.version_info < (3, 0):
        a_string = a_string.decode('utf-8').encode('unicode-escape')
    return a_string.replace('\u014d', 'o').replace('\u014c', 'O')

def find_location(row, prefecture=True):
    for (col, store_info) in [('air_area_name', air_store_info), ('hpg_area_name', hpg_store_info)]:
        if row['coordinates'] in store_info['coordinates'].values:
            area = clean_string(store_info.loc[store_info['coordinates'] == row['coordinates']][col].values[0])
            return area.split(' ')[0].split('-')[0] if prefecture is True else area
            
combined_df_20 = combined_coverage_dfs['20'].copy()
combined_df_20['prefecture'] = combined_df_20.apply(find_location, axis=1)
combined_df_20['area'] = combined_df_20.apply(lambda _: find_location(_, prefecture=False), axis=1)

display(combined_df_20.head())
sorted_df_20 = combined_df_20.copy()
sorted_df_20.sort_values(by=['prefecture', 'area'], axis=0, inplace=True)
sorted_df_20.reset_index(drop=False, inplace=True)

def find_changes(df, target_col, handler=None):
    indexes, values, previous_value = [], [], None
    for i, row in df.iterrows():
        current_value = row[target_col] if handler is None else handler(row[target_col])
        if previous_value != current_value:
            previous_value = current_value
            indexes.append(i)
            values.append(current_value)
                
    return indexes, values

(prefecture_indexes, prefecture_values) = find_changes(sorted_df_20, 'prefecture')
(city_indexes, city_values) = find_changes(sorted_df_20, 'area', handler=lambda _: ' '.join(_.split(' ')[:2]))
(area_indexes, area_values) = find_changes(sorted_df_20, 'area')

print('Number of prefecture changes: {}'.format(len(prefecture_indexes)))
print('Number of city changes:       {}'.format(len(city_indexes)))
print('Number of area changes:       {}'.format(len(area_indexes)))
previous_df = combined_coverage_dfs['20'].drop(['coordinates'], axis=1)
display_df = sorted_df_20.drop(['index', 'coordinates', 'prefecture', 'area'], axis=1)

fig = figure(figsize=(12, 15))
gs = gridspec.GridSpec(19, 2)

cbar_ax = subplot(gs[0, :])
ax1, ax2 = subplot(gs[1:10, 0]), subplot(gs[1:10, 1])
ax3, ax4 = subplot(gs[10:, 0]), subplot(gs[10:, 1])

cbar_kwargs = dict(cbar=True, cbar_ax=cbar_ax, cbar_kws=dict(orientation='horizontal'))
sns.heatmap(previous_df, ax=ax1, xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(display_df, ax=ax2, xticklabels=False, yticklabels=False, cbar=False)
sns.heatmap(display_df, ax=ax3, yticklabels=False, cbar=False)
sns.heatmap(display_df, ax=ax4, yticklabels=False, **cbar_kwargs)

_ = ax1.set_title('1) Where We Left Off {}'.format(previous_df.shape))
_ = ax2.set_title('2) Sorted - No Lines {}'.format(display_df.shape))
_ = ax3.set_title('3) Sorted - City Lines {}'.format(display_df.shape))
_ = ax4.set_title('4) Sorted - Prefecture Lines {}'.format(display_df.shape))

ax3.hlines(city_indexes, *ax3.get_xlim(), colors='#00e600')
ax4.hlines(prefecture_indexes, *ax4.get_xlim(), colors='#00e600')

for i, index_update in enumerate(prefecture_indexes):
    ax4.text(
        x=ax4.get_xlim()[1] + 0.5,
        y=index_update - 15 if i % 2 == 1 else index_update + 15,
        s='{} - {}'.format(index_update, prefecture_values[i]),
        ha='left',
        va='top' if i % 2 == 1 else 'bottom',
        fontsize=8
    )

plt.tight_layout()
plt.show()
air_info_copy, hpg_info_copy = air_store_info.copy(), hpg_store_info.copy()

air_info_copy['coordinate_count'] = air_info_copy.groupby(
    ['latitude', 'longitude']
).latitude.transform('count').astype(int)

hpg_info_copy['coordinate_count'] = hpg_info_copy.groupby(
    ['latitude', 'longitude']
).latitude.transform('count').astype(int)
sns.distplot(air_info_copy['station_vincenty'], rug=True, kde=True)
plt.title('AIR - Distances (km) to Stations Distribution')
plt.show()

sns.distplot(hpg_info_copy['station_vincenty'], rug=True, kde=True)
plt.title('HPG - Distances (km) to Stations Distribution')
plt.show()
p = sns.jointplot(x='station_vincenty', y='coordinate_count', data=air_info_copy, kind='kde')
plt.title('AIR KDE Joint Plot', loc='left')
p.plot_joint(plt.scatter, c='r', s=30, linewidth=1, marker='x')
plt.show()

p = sns.jointplot(x='station_vincenty', y='coordinate_count', data=hpg_info_copy, kind='kde')
plt.title('HPG KDE Joint Plot', loc='left')
p.plot_joint(plt.scatter, c='r', s=30, linewidth=1, marker='x')
plt.show()
def view_distances(df, distance, cols=['station_vincenty', 'coordinate_count']):
    return df[cols].groupby(cols).filter(
         lambda _: _[cols[0]].mean() > distance
    ).drop_duplicates().sort_values(by=cols).reset_index(drop=True)

display(view_distances(air_info_copy, 8))
display(view_distances(hpg_info_copy, 10))
weather_stations['date_terminated'] = pd.to_datetime(weather_stations['date_terminated'], format='%Y-%m-%d').dt.date
active_stations = weather_stations.loc[pd.isnull(weather_stations['date_terminated'])]

def calculate_cumulative_opacity(base_opacity, num_stacks):
    cumulative_opacity = base_opacity
    for i in range(num_stacks - 1):
        cumulative_opacity = cumulative_opacity + (1 - cumulative_opacity) * base_opacity
    return cumulative_opacity


def coord_groups(df):
    unique_groups = df.groupby(['latitude', 'longitude']).groups
    unique_coords = unique_groups.keys()
    unique_vals = [unique_groups[_] for _ in unique_coords]
    return (unique_groups, unique_coords, unique_vals)

(unique_air_groups, unique_air_coords, unique_air_vals) = coord_groups(air_store_info)
(unique_hpg_groups, unique_hpg_coords, unique_hpg_vals) = coord_groups(hpg_store_info)
(unique_active_groups, unique_active_coords, unique_active_vals) = coord_groups(active_stations)

base_opacity = 0.3
air_opacity_vals = [calculate_cumulative_opacity(base_opacity, len(_)) for _ in unique_air_vals]
hpg_opacity_vals = [calculate_cumulative_opacity(base_opacity, len(_)) for _ in unique_hpg_vals]
###############################################
# Initialize Map
###############################################
f_map = folium.Map(location=[38, 137], zoom_start=5, tiles='Cartodb Positron')

###############################################
# Create Active Weather Station MarkerCluster
###############################################
marker_cluster = folium_plugins.MarkerCluster(name='marker_cluster', control=True, overlay=True)

for i, coords in enumerate(unique_active_coords):
    _station = active_stations.loc[unique_active_vals[i][0]]

    marker = folium.Marker(
        location=coords, 
        popup=folium.Popup('ID:  {}<br>Coords:  {}'.format(
            _station['id'],
            [float('{:.4f}'.format(_)) for _ in coords]
        ))
    )

    marker_cluster.add_child(marker)

###############################################
# Create AIR Stores FeatureGroup
###############################################
air_stores_group = folium.FeatureGroup('air_stores')

for i, coords in enumerate(unique_air_coords):
    _store_group = air_store_info.loc[unique_air_vals[i][0]]

    air_stores_group.add_child(folium.RegularPolygonMarker(
        location=coords,
        popup='Coords:  {}<br>Store Count:  {}<br>Station:  {}<br>Vincenty:  {:.5f}'.format(
            coords,
            len(unique_air_vals[i]),
            _store_group['station_id'],
            _store_group['station_vincenty']
        ),
        fill_opacity=air_opacity_vals[i],
        fill_color='red',
        number_of_sides=100, weight=0, radius=10
    ))

###############################################
# Create HPG Stores FeatureGroup
###############################################
hpg_stores_group = folium.FeatureGroup('hpg_stores')

for i, coords in enumerate(unique_hpg_coords):
    _store_group = hpg_store_info.loc[unique_hpg_vals[i][0]]

    hpg_stores_group.add_child(folium.RegularPolygonMarker(
        location=coords,
        popup='Coords:  {}<br>Store Count:  {}<br>Station:  {}<br>Vincenty:  {:.5f}'.format(
            coords,
            len(unique_hpg_vals[i]),
            _store_group['station_id'],
            _store_group['station_vincenty']
        ),
        fill_opacity=hpg_opacity_vals[i],
        fill_color='green',
        number_of_sides=100, weight=0, radius=10
    ))

###############################################
# Add Active Stations FeatureGroup
# Add HPG and AIR Stores FeatureGroups
###############################################
active_stations_group = folium.FeatureGroup(name='active_stations')
active_stations_group.add_child(marker_cluster)
f_map.add_child(active_stations_group)

f_map.add_child(hpg_stores_group)
f_map.add_child(air_stores_group)

###############################################
# Add Map Extras and Display Map
###############################################
f_map.add_child(folium.map.LayerControl(collapsed=False))
# f_map.add_child(features.LatLngPopup())
f_map.add_child(folium_plugins.MeasureControl(
    position='bottomleft', primary_length_unit='kilometers', secondary_length_unit='miles'
))

display(f_map)
store_id_relation = pd.read_csv('{}/store_id_relation.csv'.format(original_data_dir))

temp_air_store_info = air_store_info.loc[:, ['air_store_id', 'latitude', 'longitude']]
temp_air_store_info.columns = ['air_store_id', 'air_latitude', 'air_longitude']

temp_hpg_store_info = hpg_store_info.loc[:, ['hpg_store_id', 'latitude', 'longitude']]
temp_hpg_store_info.columns = ['hpg_store_id', 'hpg_latitude', 'hpg_longitude']

identical_stores_df = pd.merge(store_id_relation, temp_air_store_info, on=['air_store_id'], how='inner')
identical_stores_df = pd.merge(identical_stores_df, temp_hpg_store_info, on=['hpg_store_id'], how='inner')

identical_stores_df['vincenty'] = identical_stores_df.apply(
    lambda _: vincenty((_['air_latitude'], _['air_longitude']), (_['hpg_latitude'], _['hpg_longitude'])).km,
    axis=1
)
print(identical_stores_df.shape)
display(identical_stores_df.head())
sns.distplot(identical_stores_df['vincenty'].values, rug=True, kde=False)
plt.show()
