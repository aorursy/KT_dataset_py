import geopandas as gpd #see this for installing geopandas https://www.youtube.com/watch?v=LNPETGKAe0c

import numpy as np

import pandas as pd

import folium

from folium.plugins import TimeSliderChoropleth

import branca.colormap as cm
corona_df=pd.read_csv("/kaggle/input/covid_data_til_19_May.csv")

corona_df.head()
country = gpd.read_file("/kaggle/input/bangladesh.json")
country=country[["id","NAME_2","geometry"]]
country["geometry"] = country["geometry"].simplify(0.01, preserve_topology = False)

country.head()
country = country.rename(columns={'NAME_2': 'Zilla'})

country.head()
def correct_date(date_str):

    list_dates = date_str.split("/")

    day = list_dates[0]

    month = list_dates[1]

    year = list_dates[2]

    

    if len(day) == 1:

        day = "0" + day

    if len(month) == 1:

        month = "0" + month

        

    return "/".join([day, month, year])
corona_df["Date"] = corona_df["Date"].apply(correct_date)
corona_df = corona_df[corona_df.Cases != 0]
sorted_df = corona_df.sort_values(['Zilla', 

                     'Date']).reset_index(drop=True)
sum_df = sorted_df.groupby(['Zilla', 'Date'], as_index=False).sum()
joined_df = sum_df.merge(country, on='Zilla')
joined_df['log_Confirmed'] = np.log10(joined_df['Cases'])
joined_df['date_sec'] = pd.to_datetime(joined_df['Date'],format='%d/%m/%Y')

joined_df['date_sec']=[(joined_df['date_sec'][i]).timestamp() for i in range(0,joined_df['date_sec'].shape[0])]

joined_df['date_sec'] = joined_df['date_sec'].astype(int).astype(str)
joined_df = joined_df[['Zilla','Date', 'date_sec', 'log_Confirmed', 'geometry']]
max_colour = max(joined_df['log_Confirmed'])

min_colour = min(joined_df['log_Confirmed'])

cmap = cm.linear.OrRd_09.scale(min_colour, max_colour)

joined_df['colour'] = joined_df['log_Confirmed'].map(cmap)
zilla_list = joined_df['Zilla'].unique().tolist()

zilla_idx = range(len(zilla_list))

style_dict = {}

for i in zilla_idx:

    zilla = zilla_list[i]

    result = joined_df[joined_df['Zilla'] == zilla]

    inner_dict = {}

    for _, r in result.iterrows():

        inner_dict[r['date_sec']] = {'color': r['colour'], 'opacity': 0.8}

    style_dict[str(i)] = inner_dict
zilla_df = joined_df[['geometry']]

zilla_gdf = gpd.GeoDataFrame(zilla_df)

zilla_gdf = zilla_gdf.drop_duplicates().reset_index()
slider_map = folium.Map(location = (23.6850, 90.3563), zoom_start = 6.5,tiles='cartodbpositron')



_ = TimeSliderChoropleth(

    data=zilla_gdf.to_json(),

    styledict=style_dict,



).add_to(slider_map)



_ = cmap.add_to(slider_map)

cmap.caption = "Log of number of confirmed cases"
slider_map
#slider_map.save(outfile='Time_mapping.html')