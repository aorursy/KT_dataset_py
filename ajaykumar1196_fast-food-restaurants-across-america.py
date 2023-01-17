import numpy as np
import pandas as pd
import gmplot
from fuzzywuzzy import process, fuzz
data = pd.read_csv('../input/FastFoodRestaurants.csv')
data = data[["address", "city", "country", "latitude", "longitude", "name", "postalCode", "province"]]
data.head()
sorted(data.name.unique())
data['lowername'] = data['name'].apply(lambda x : x.lower().strip())
unique_names = sorted(data.lowername.unique())
restaurants_counts = data.lowername.value_counts()
restaurants_counts = restaurants_counts[restaurants_counts>250]
restaurants_list = list(restaurants_counts.index)
restaurants_list
def replace_name(data, column_name, brand, threshold_ratio = 90):
    query = data[column_name].unique()
    results = process.extract(brand, query, limit=10, scorer=fuzz.token_sort_ratio)
    string_matches = [results[0] for results in results if results[1] >= threshold_ratio]
    rows_with_matches = data[column_name].isin(string_matches) 
    data.loc[rows_with_matches, column_name] = brand
    return data.copy()
similar_name_list = list()
for restuarant in restaurants_list:
    query = data['lowername'].unique()
    results = process.extract(restuarant, query, limit=9, scorer=fuzz.token_sort_ratio)
    similar_name_list.append(results)
similar_name_list

##Check the token_sort_ratio upto the closed value.
#mcdonald's - 58
#burger king - 100
#wendy's - 77
tokens = [57, 100, 100, 76, 90, 37, 50, 88]
for token, restuarant in zip(tokens, restaurants_list):
    replace_name(data,'lowername',restuarant, token)
data = replace_name(data,'lowername','kentucky fried chicken', 90)
data.loc[data.lowername.str.startswith('kentucky fried chicken'), 'lowername'] = 'kfc'
sorted(data.lowername.unique())
data_top = data[data.lowername.isin(restaurants_list)]
data_top.lowername.value_counts()
gmap = gmplot.GoogleMapPlotter.from_geocode('US',5)
#Then generate a heatmap using the latitudes and longitudes
gmap.heatmap(data['latitude'], data['longitude'])
gmap.draw('full_list.html')
gmap = gmplot.GoogleMapPlotter.from_geocode('US',5)
#Then generate a heatmap using the latitudes and longitudes
gmap.heatmap(data_top['latitude'], data_top['longitude'])
gmap.draw('top_list.html')
data_top = data_top[["country", "province", "city", "lowername"]]
%matplotlib inline
data_top_province = data_top.province.value_counts()
data_top_province_list = list(data_top_province[data_top_province>200].index)
data_top_province_top = data_top[data_top.province.isin(data_top_province_list)]
data_top_province_top_group = data_top_province_top.groupby(['province','lowername'])
data_top_province_top_group.size().unstack().plot(kind ='bar', figsize =(16,7))