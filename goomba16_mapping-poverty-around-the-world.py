import numpy as np 
import pandas as pd 
import folium
from folium.plugins import HeatMap
import seaborn as sns
from functools import reduce
import geopandas as gpd
kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
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
mpi_national = pd.read_csv('../input/mpi/MPI_national.csv')
lat_long_info = pd.read_csv('../input/world-countries-and-continents-details/Countries Longitude and Latitude.csv')
countries = gpd.read_file('../input/countries-shape-files/ne_10m_admin_0_countries.shp')
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
def rename_df_columns(df, current_names, new_names):
    assert len(current_names) == len(new_names)
    columns = {key: new_names[i] for i, key in enumerate(current_names)}
    return df.rename(index=str, columns=columns)

def merge_dfs_on_column(dfs, column_name='Country'):
    return reduce(lambda left,right: pd.merge(left,right,on=column_name), dfs)

# helper function to shift the lat and long coords of markers to prevent positioning on top of each other
def shift_coords(df, amount_coord1, amount_coord2):
    df['latitude'] = df['latitude'] + amount_coord1
    df['longitude'] = df['longitude'] + amount_coord2
    return df

def update_country_names(df, mappings):
    # Update the country names before merge to avoid losing data
    for key in mappings.keys():
        df.loc[df.Country == key, 'Country'] = mappings[key]
    return df
    
def update_country_profile_with_coords(profile_df, mappings, coords_df, shift=True, amount_coord1=0, amount_coord2=0.5):
    profile_df = update_country_names(profile_df, mappings)  
    coords_df = shift_coords(coords_df, amount_coord1, amount_coord2) if shift else coords_df
    profile_updated = merge_dfs_on_column([coords_df, profile_df])
    return profile_updated

mpi_national['MPI Diff'] = mpi_national['MPI Rural'] - mpi_national['MPI Urban']
lat_long_info = rename_df_columns(lat_long_info, ['name'], ['Country'])
mpi_national_updated_a = update_country_profile_with_coords(mpi_national, country_mappings_a, lat_long_info, shift=False)
create_coords_column(mpi_national_updated_a)

countries = rename_df_columns(countries, ['NAME'], ['Country'])
countries.geometry = countries.geometry.simplify(0.3)
mpi_national_updated_b = update_country_profile_with_coords(mpi_national, country_mappings_b, countries, shift=False)

lowest_MPI_urban = get_coords_of_n_smallest(mpi_national_updated_a, metric='MPI Urban')
highest_MPI_urban = get_coords_of_n_largest(mpi_national_updated_a, metric='MPI Urban')
lowest_MPI_rural = get_coords_of_n_smallest(mpi_national_updated_a, metric='MPI Rural')
highest_MPI_rural = get_coords_of_n_largest(mpi_national_updated_a, metric='MPI Rural')
lowest_MPI_diff = get_coords_of_n_smallest(mpi_national_updated_a, metric='MPI Diff')
highest_MPI_diff = get_coords_of_n_largest(mpi_national_updated_a, metric='MPI Diff')
base_layer = countries.geometry.to_json()
mpi_layer = mpi_national_updated_b[['Country', 'geometry']].to_json()
style_function = lambda x: {'fillColor': '#ffffff', 'color': '#000000', 'weight' : 1}
urban_map = folium.Map(location=[10, 0], zoom_start=2)
folium.GeoJson(
    base_layer,
    name='base',
    style_function=style_function
).add_to(urban_map)

urban_map.choropleth(
    geo_data=mpi_layer,
    name='mpi urban choropleth',
    key_on='properties.Country',
    fill_color='YlOrRd',
    data=mpi_national_updated_b,
    columns=['Country', 'MPI Urban'],
    legend_name='MPI Urban'
)

folium.LayerControl().add_to(urban_map)
add_markers_to_map(urban_map, lowest_MPI_urban, metric='MPI Urban')
add_markers_to_map(urban_map, highest_MPI_urban, metric='MPI Urban', color='red')
urban_map
rural_map = folium.Map(location=[10, 0], zoom_start=2)
folium.GeoJson(
    base_layer,
    name='base',
    style_function=style_function
).add_to(rural_map)
rural_map.choropleth(
    geo_data=mpi_layer,
    name='mpi rural choropleth',
    key_on='properties.Country',
    fill_color='YlOrRd',
    data=mpi_national_updated_b,
    columns=['Country', 'MPI Rural'],
    legend_name='MPI Rural'
)

folium.LayerControl().add_to(rural_map)
add_markers_to_map(rural_map, lowest_MPI_rural, metric='MPI Rural')
add_markers_to_map(rural_map, highest_MPI_rural, metric='MPI Rural', color='red')
rural_map
diff_map = folium.Map(location=[10, 0], zoom_start=2)
folium.GeoJson(
    base_layer,
    name='base',
    style_function=style_function
).add_to(diff_map)

diff_map.choropleth(
    geo_data=mpi_layer,
    name='mpi diff choropleth',
    key_on='properties.Country',
    fill_color='YlOrRd',
    data=mpi_national_updated_b,
    columns=['Country', 'MPI Diff'],
    legend_name='MPI Diff'
)

folium.LayerControl().add_to(diff_map)
add_markers_to_map(diff_map, lowest_MPI_diff, metric='MPI Diff')
add_markers_to_map(diff_map, highest_MPI_diff, metric='MPI Diff', color='red')
diff_map
pov_metrics = mpi_national_updated_a[['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban', 'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural']]
corr = pov_metrics.corr()
sns.heatmap(corr)
health_nutr_pop = pd.read_csv('../input/health-nutrition-and-population-statistics/data.csv')
def extract_data_for_indicator(df, ind_name, years):
    return df[df['Indicator Name'] == ind_name][[*years, 'Country Name']]

def generate_new_column_names(ind_name, years):
    return ['{} - {}'.format(ind_name, year) for year in years]

def create_indicator_df(df, ind_name, years):
    new_df = extract_data_for_indicator(df, ind_name, years)
    return rename_df_columns(new_df, [*years, 'Country Name'], [*generate_new_column_names(ind_name, years), 'Country'])

def create_indicator_dfs(df, ind_names, years_arr):
    return [create_indicator_df(df, ind_name, years_arr[i]) for i, ind_name in enumerate(ind_names)]
    
def calc_rank(df, col_name, ascending):
    df[col_name] = df[col_name].rank(ascending=ascending)/df[col_name].count()

def calc_ranks(df, ind_name, years, ascending):
    col_names = generate_new_column_names(ind_name, years)
    for col_name in col_names:
        calc_rank(df, col_name, ascending)

def calc_all_ranks(df, ind_names, years_arr, sort_order_arr):
    for i, ind_name in enumerate(ind_names):
        calc_ranks(df, ind_name, years_arr[i], sort_order_arr[i])
        
def calc_final_rank(df, rank_name, total_ind_col='Total Indicators'):
    cols = list(df)
    cols.remove(total_ind_col)
    df[rank_name] = df[cols].sum(axis=1)/df[total_ind_col]

def create_country_profile(df, ind_names, years_arr):
    # Combine all of the metrics under single profile
    profile = pd.DataFrame({'Country': df['Country Name'].unique()})
    profile = merge_dfs_on_column([profile, *create_indicator_dfs(df, ind_names, years_arr)])
    profile['Total Indicators'] = profile.count(axis=1)-1
    # Filter out countries/regions for which we have no information in any of the categories or we have information in only one of the categories
    return profile[profile['Total Indicators'] > 1]
ONE_YEAR = ['2012']
TWO_YEARS = ['2012','2013']
THREE_YEARS = ['2012','2013', '2014']
FOUR_YEARS = ['2012','2013', '2014', '2015']
RANK_ORDER_ASCENDING = [True, False, False, False, True, True, True, True, True, False, False, False]
INDICATOR_NAMES = [
    'Adolescent fertility rate (births per 1,000 women ages 15-19)',
    'Condom use, population ages 15-24, female (% of females ages 15-24)',
    'Condom use, population ages 15-24, male (% of males ages 15-24)',
    'Contraceptive prevalence, modern methods (% of women ages 15-49)',
    'Fertility rate, total (births per woman)',
    'Population growth (annual %)',
    'Rural population growth (annual %)',
    'Urban population growth (annual %)',
    'Birth rate, crude (per 1,000 people)',
    'Condom use with non regular partner, % adults(15-49), female',
    'Condom use with non regular partner, % adults(15-49), male',
    'Demand for family planning satisfied by modern methods (% of married women with demand for family planning)'
]
RECENT_YEARS_WITH_DATA = [
    THREE_YEARS,
    TWO_YEARS,
    TWO_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS
]

# Create profile
country_population_rise_profile = create_country_profile(health_nutr_pop, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA)

# Turn numeric values into ranks 
calc_all_ranks(country_population_rise_profile, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA, RANK_ORDER_ASCENDING)

# Calculate the 'Poor access to healthcare rank'
calc_final_rank(country_population_rise_profile, 'Poor population growth control score')
country_population_rise_profile.sort_values(by=['Poor population growth control score'], ascending=False)[['Country', 'Poor population growth control score', 'Total Indicators']]
country_population_rise_profile_updated = update_country_profile_with_coords(country_population_rise_profile, country_mappings_a, lat_long_info)
create_coords_column(country_population_rise_profile_updated)
lowest_rank = get_coords_of_n_smallest(country_population_rise_profile_updated, metric='Poor population growth control score')
highest_rank = get_coords_of_n_largest(country_population_rise_profile_updated, metric='Poor population growth control score')
add_markers_to_map(urban_map, lowest_rank,  metric='Poor population growth control score', color='blue', icon='child', prefix='fa')
add_markers_to_map(urban_map, highest_rank,  metric='Poor population growth control score', color='red', icon='child', prefix='fa')
urban_map
RANK_ORDER_ASCENDING = [False, False, False, False, False, True, False, False, True, False, True, True, True, True]
INDICATOR_NAMES = [
    'Hospital beds (per 1,000 people)',
    'Physicians (per 1,000 people)',
    'Specialist surgical workforce (per 100,000 population)',
    'Number of surgical procedures (per 100,000 population)',
    'Births attended by skilled health staff (% of total)',
    'External resources for health (% of total expenditure on health)',
    'Health expenditure per capita (current US$)',
    'Health expenditure, total (% of GDP)',
    'Lifetime risk of maternal death (%)',
    'Nurses and midwives (per 1,000 people)',
    'Risk of impoverishing expenditure for surgical care (% of people at risk)',
    'Risk of catastrophic expenditure for surgical care (% of people at risk)',
    'Out-of-pocket health expenditure (% of total expenditure on health)',
    'Health expenditure, private (% of total health expenditure)'
]
RECENT_YEARS_WITH_DATA = [
    ONE_YEAR,
    TWO_YEARS,
    THREE_YEARS,
    ONE_YEAR,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    FOUR_YEARS,
    TWO_YEARS,
    ['2014'],
    ['2014'],
    THREE_YEARS,
    THREE_YEARS
]

# Create profile
country_health_access_profile = create_country_profile(health_nutr_pop, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA)

# Turn numeric values into ranks 
calc_all_ranks(country_health_access_profile, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA, RANK_ORDER_ASCENDING)

# Calculate the 'Poor access to healthcare rank'
calc_final_rank(country_health_access_profile, 'Impeded access to healthcare score')
country_health_access_profile.sort_values(by=['Impeded access to healthcare score'], ascending=False)[['Country', 'Impeded access to healthcare score', 'Total Indicators']]
country_health_access_profile_updated = update_country_profile_with_coords(country_health_access_profile, country_mappings_a, lat_long_info)
create_coords_column(country_health_access_profile_updated)
lowest_rank = get_coords_of_n_smallest(country_health_access_profile_updated, metric='Impeded access to healthcare score')
highest_rank = get_coords_of_n_largest(country_health_access_profile_updated, metric='Impeded access to healthcare score')
add_markers_to_map(urban_map, lowest_rank, metric='Impeded access to healthcare score', color='green', icon='ambulance', prefix='fa')
add_markers_to_map(urban_map, highest_rank, metric='Impeded access to healthcare score', color='red', icon='ambulance', prefix='fa')
urban_map 
RANK_ORDER_ASCENDING = [False, False, False, False, False, False, True, True, True]
INDICATOR_NAMES = [
    'Improved sanitation facilities (% of population with access)',
    'Improved sanitation facilities, rural (% of rural population with access)',
    'Improved sanitation facilities, urban (% of urban population with access)',
    'Improved water source (% of population with access)',
    'Improved water source, rural (% of rural population with access)',
    'Improved water source, urban (% of urban population with access)',
    'People practicing open defecation (% of population)',
    'People practicing open defecation, rural (% of rural population)',
    'People practicing open defecation, urban (% of urban population)'
]
RECENT_YEARS_WITH_DATA = [
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS,
    FOUR_YEARS
]

# Create profile
country_sanitation_profile = create_country_profile(health_nutr_pop, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA)

# Turn numeric values into ranks 
calc_all_ranks(country_sanitation_profile, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA, RANK_ORDER_ASCENDING)

# Calculate the 'Poor sanitary conditions rank'
calc_final_rank(country_sanitation_profile, 'Poor sanitary conditions score')
country_sanitation_profile.sort_values(by=['Poor sanitary conditions score'], ascending=False)[['Country', 'Poor sanitary conditions score', 'Total Indicators']]
country_sanitation_profile_updated = update_country_profile_with_coords(country_sanitation_profile, country_mappings_a, lat_long_info, amount_coord1=-1, amount_coord2=-0.5)
create_coords_column(country_sanitation_profile_updated)
lowest_rank = get_coords_of_n_smallest(country_sanitation_profile_updated, metric='Poor sanitary conditions score')
highest_rank = get_coords_of_n_largest(country_sanitation_profile_updated, metric='Poor sanitary conditions score')
add_markers_to_map(urban_map, lowest_rank, metric='Poor sanitary conditions score', color='green', icon='tint', prefix='fa')
add_markers_to_map(urban_map, highest_rank, metric='Poor sanitary conditions score', color='red', icon='tint', prefix='fa')
urban_map
RANK_ORDER_ASCENDING = [True, True, True, True, True, True]
INDICATOR_NAMES = [
    'Malnutrition prevalence, height for age (% of children under 5)',
    'Malnutrition prevalence, weight for age (% of children under 5)',
    'Number of people who are undernourished',
    'Prevalence of severe wasting, weight for height (% of children under 5)',
    'Prevalence of wasting (% of children under 5)',
    'Prevalence of undernourishment (% of population)'
]
RECENT_YEARS_WITH_DATA = [
    THREE_YEARS,
    THREE_YEARS,
    FOUR_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    FOUR_YEARS
]

# Create profile
country_malnourishment_profile = create_country_profile(health_nutr_pop, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA)

# Turn numeric values into ranks 
calc_all_ranks(country_malnourishment_profile, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA, RANK_ORDER_ASCENDING)

# Calculate the 'Malnourishment rank'
calc_final_rank(country_malnourishment_profile, 'Malnourishment score')
country_malnourishment_profile.sort_values(by=['Malnourishment score'], ascending=False)[['Country', 'Malnourishment score', 'Total Indicators']]
country_malnourishment_profile_updated = update_country_profile_with_coords(country_malnourishment_profile, country_mappings_a, lat_long_info, amount_coord1=0.5, amount_coord2=-0.5)
create_coords_column(country_malnourishment_profile_updated)
lowest_rank = get_coords_of_n_smallest(country_malnourishment_profile_updated, metric='Malnourishment score')
highest_rank = get_coords_of_n_largest(country_malnourishment_profile_updated, metric='Malnourishment score')
add_markers_to_map(urban_map, lowest_rank, metric='Malnourishment score', color='green', icon='apple', prefix='fa')
add_markers_to_map(urban_map, highest_rank, metric='Malnourishment score', color='red', icon='apple', prefix='fa')
urban_map
RANK_ORDER_ASCENDING = [False, False, False, False, False, False, False]
INDICATOR_NAMES = [
    'Literacy rate, adult total (% of people ages 15 and above)',
    'Literacy rate, youth total (% of people ages 15-24)',
    'Primary completion rate, total (% of relevant age group)',
    'Public spending on education, total (% of GDP)',
    'School enrollment, primary (% net)',
    'School enrollment, secondary (% net)',
    'School enrollment, tertiary (% gross)'
]
RECENT_YEARS_WITH_DATA = [
    FOUR_YEARS,
    FOUR_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS,
    THREE_YEARS
]

# Create profile
country_education_profile = create_country_profile(health_nutr_pop, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA)

# Turn numeric values into ranks 
calc_all_ranks(country_education_profile, INDICATOR_NAMES, RECENT_YEARS_WITH_DATA, RANK_ORDER_ASCENDING)

# Calculate the 'Low ed rank'
calc_final_rank(country_education_profile, 'Impdeded Access to Education Score')
country_education_profile.sort_values(by=['Impdeded Access to Education Score'], ascending=False)[['Country', 'Impdeded Access to Education Score', 'Total Indicators']]
country_education_profile_updated = update_country_profile_with_coords(country_education_profile, country_mappings_a, lat_long_info, amount_coord1=-0.5, amount_coord2=-0.5)
create_coords_column(country_education_profile_updated)
lowest_rank = get_coords_of_n_smallest(country_education_profile_updated, metric='Impdeded Access to Education Score')
highest_rank = get_coords_of_n_largest(country_education_profile_updated, metric='Impdeded Access to Education Score')
add_markers_to_map(urban_map, lowest_rank, metric='Impdeded Access to Education Score', color='green', icon='laptop', prefix='fa')
add_markers_to_map(urban_map, highest_rank, metric='Impdeded Access to Education Score', color='red', icon='laptop', prefix='fa')
urban_map
gpi = pd.read_csv('../input/gpi2008-2016/gpi_2008-2016.csv')
country_mappings = {
    'United Kingdom': 'UK',
    'United States': 'US',
    'Ivory Coast': 'Côte d’Ivoire',
    'Democratic Republic of the Congo': 'Congo - Kinshasa',
    'Republic of the Congo': 'Congo - Brazzaville'
}
gpi = rename_df_columns(gpi, ['country'], ['Country'])
gpi = update_country_profile_with_coords(gpi, country_mappings, lat_long_info)
create_coords_column(gpi)
gpi['av_2012_2015'] = gpi[['score_2012', 'score_2013', 'score_2014', 'score_2015']].mean(axis=1)
lowest_score = get_coords_of_n_smallest(gpi, metric='av_2012_2015')
highest_score = get_coords_of_n_largest(gpi, metric='av_2012_2015')
add_markers_to_map(urban_map, lowest_score, metric='global peace index', color='lightgreen', icon='bomb', prefix='fa')
add_markers_to_map(urban_map, highest_score, metric='global peace index', color='purple', icon='bomb', prefix='fa')
urban_map 
whr_2015 = pd.read_csv('../input/world-happiness/2015.csv')
country_mappings = {
    'United Kingdom': 'UK',
    'United States': 'US',
    'Ivory Coast': 'Côte d’Ivoire',
    'Congo (Kinshasa)': 'Congo - Kinshasa',
    'Congo (Brazzaville)': 'Congo - Brazzaville',
    'Palestinian Territories': 'Palestine'
}
whi = update_country_profile_with_coords(whr_2015, country_mappings, lat_long_info, amount_coord1=-0.5, amount_coord2=0)
create_coords_column(whi)
lowest_rank = get_coords_of_n_smallest(whi, metric='Happiness Rank')
highest_rank = get_coords_of_n_largest(whi, metric='Happiness Rank')
add_markers_to_map(urban_map, lowest_rank, metric='Happiness Rank', color='darkgreen', icon='thumbs-up')
add_markers_to_map(urban_map, highest_rank, metric='Happiness Rank', color='darkpurple', icon='thumbs-down')
urban_map