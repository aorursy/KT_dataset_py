# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import altair as alt
from altair import datum
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
alt.renderers.enable('notebook')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from IPython.display import HTML
# from altair.vega import v3
import json
##-----------------------------------------------------------
# in altair-2.2.0, v3.SCHEMA_VERSION is no longer available hence
v3_SCHEMA_VERSION = 'v3.3.1'
# This whole section 
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3_SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "Due to Altair rendering issues in Kaggle, we use some rendering functions from "
    "<a href=https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey>notsluh's notebook</a> to fix this "
)))
# load datasets and convert all column names to lower case

hfacilities = pd.read_csv("../input/ghana-health-facilities/health-facilities-gh.csv")
hfacilities.columns = hfacilities.columns.str.lower()
hfacilities = hfacilities.rename(columns={'facilityname': 'facility_name'})

regions = pd.read_csv('../input/additional/region_stats.csv')
regions.columns = regions.columns.str.lower()

map_regions = gpd.GeoDataFrame.from_file('../input/additional/gadm36_GHA_1.shp')
map_regions = map_regions[['NAME_1', 'geometry']]
map_regions = map_regions.replace('', np.NaN, regex=True)

map_df = map_regions.rename(columns={'NAME_1': 'region', 'NAME_2': 'district', 'TYPE_2': 'type'})
map_df = map_df.merge(regions, on='region')

del regions
c = alt.Chart(hfacilities).mark_bar().encode(
    y='type',
    x='count()',
    tooltip='count()',
).properties(
    title="Distribution of facility type"
)

# you can do without the render function outside kaggle. This was the only way to display the chart
render(c)
hfacilities[hfacilities.type == 'Centre']
hfacilities.loc[hfacilities['type'] == 'Centre', 'type'] = "Health Centre"
hfacilities.loc[hfacilities['type'] == 'CPHS', 'type'] = 'CHPS'
hfacilities.loc[hfacilities['type'] == 'clinic', 'type'] = 'Clinic'
hfacilities.loc[hfacilities['type'] == 'DHD', 'type'] = 'District Health Directorate'
hfacilities.loc[hfacilities['type'] == 'Municipal  Health Directorate', 'type'] = 'Municipal Health Directorate'
base = alt.Chart(hfacilities).mark_bar().encode(
    x='type',
    tooltip='count()',
)

a = base.encode(
    alt.Y('count()')
).properties(
    title="Distribution of facility type"
)

b = base.encode(
    y=alt.Y('count()', scale=alt.Scale(type='log'))
).properties(
    title="Distribution of facility type on a log scale"
)

# you can do without the render function outside kaggle. This was the only way to display the chart
render(a | b)
c = alt.Chart(hfacilities).mark_bar().encode(
    y='district',
    x='count()'
).transform_filter(
    datum.type == 'Others'
)

render(c)
# hfacilities.iloc[2617]
hfacilities.loc[hfacilities.district == 'Kassena-Nankana West']
base = alt.Chart(hfacilities).mark_bar().transform_filter(datum.type == 'Others').properties(width=300, height=100)

a = base.encode(
    alt.Y('region'),
    alt.X('count()'),
    alt.Tooltip('count()')
)

b = base.encode(
    alt.X('district'),
    alt.Y('ownership'),
    alt.Color('count(ownership)', legend=alt.Legend(title='Number of facility ownership')),
    alt.Tooltip('count(ownership)')
)

render(a | b)
hfacilities.loc[hfacilities.district.isin(['Bongo', 'Kassena-Nankana']) & (hfacilities.type == 'Others')]
from sklearn.feature_extraction.text import CountVectorizer

# this will get the frequency of words or consecutive words(n-grams)
# bigrams because it's easier to understand a facility type when consecutive words are taken 
cvec = CountVectorizer(ngram_range=(2,3), stop_words='english')

other_facilities = hfacilities[hfacilities.type == 'Others'].facility_name.tolist()
cvec.fit(other_facilities)

joined = [" ".join(other_facilities)]
counts = cvec.transform(joined)

# flipped n-grams: index to index: n-grams
index_feature = {v:k for k,v in cvec.vocabulary_.items()}

n = len(cvec.vocabulary_)
c = counts.toarray()[0]

ngram_counts = [(index_feature[i], c[i]) for i in range(n)]
ngrams = filter(lambda x: x[1] > 1, ngram_counts)
ngrams = list(ngrams)

df_ngram_freq = pd.DataFrame(ngrams, columns=['ngrams', 'frequency'])

c = alt.Chart(df_ngram_freq).mark_bar().encode(
    x='frequency',
    y='ngrams',
    tooltip='frequency'
).properties(height=300)

render(c)
def check_nutrition_centre(row):
    s = row['facility_name'].lower()
    type = row['type']
    
    if ('nutrition centre' in s) or ('feeding centre' in s) or ('rehabilitation centre' in s) or ('nutrition' in s):
        return 'Feeding Centre'
    else:
        return type

hfacilities['type'] = hfacilities.apply(check_nutrition_centre, axis=1)
hfacilities.loc[hfacilities.type == 'Others']
c = alt.Chart(hfacilities).mark_bar().encode(
    y='ownership',
    x=alt.X('count()', scale=alt.Scale(type='log')),
    tooltip='count()',
).properties(
    width=500,
    height=300
)

render(c)
hfacilities.loc[hfacilities['ownership'] == 'Muslim', 'ownership'] = 'Islamic'
hfacilities.loc[hfacilities['ownership'] == 'private', 'ownership'] = 'Private'
hfacilities.loc[hfacilities['ownership'] == 'government', 'ownership'] = 'Government'
hfacilities[hfacilities.ownership == 'Mission']
hfacilities.loc[3398, 'ownership'] = 'Islamic'
hfacilities[hfacilities.ownership == 'Clinic']
c = alt.Chart(hfacilities).mark_bar().encode(
    x=alt.X('count()', scale=alt.Scale(type='log')),
    y='ownership',
    tooltip='count()',
)

render(c)
map_regions.head(1)
map_regions.plot();
map_df = map_df.assign(facility_number=hfacilities.groupby('region').count()['facility_name'].values)
map_df = map_df.assign(area_to_facility=map_df['area_km2'] / map_df['facility_number'])
map_df = map_df.assign(population_to_area= map_df['population_2016'] / map_df['area_km2'])
map_df = map_df.assign(population_to_hfacility= map_df['population_2016'] / map_df['facility_number'])
map_data = alt.InlineData(values=map_df.to_json(), #geopandas to geojson string
                          # root object type is "FeatureCollection" but we need its features
                          format=alt.DataFormat(property='features',type='json')) 

brush = alt.selection_interval()
click = alt.selection_multi(encodings=['color'])

bar = alt.Chart(hfacilities).mark_bar().encode(
    x=alt.X('count()', title='Number of Facilities'),
    y=alt.Y('region:N', sort=alt.EncodingSortField(op='count', order='descending')),
    tooltip='count()',
    color=alt.condition(click, 'region:N', alt.value('lightgray')),
).add_selection(
    click
).properties(
    width=300, 
    height=300
)

ghana = alt.Chart(map_data).mark_geoshape(
    fill='lightgray',
    stroke='white',
).encode(
    tooltip=['properties.region:N', 'properties.area_km2:Q', 'properties.population_2016:Q', 'properties.females:Q', 'properties.males:Q']
).properties(
    projection={"type":'mercator'},
    width=500,
    height=500,
    title='Distribution of Health Facilities in Ghana'
)

# Dropping facilities which have a null longitude/latitude. Fortunately, those with missing lng have missing lat.

hfacilities2 = hfacilities.dropna(subset=['longitude'])

facilities = alt.Chart(hfacilities2).mark_circle(opacity=0.5).encode( 
    longitude='longitude:Q',
    latitude='latitude:Q',
    tooltip=['facility_name', 'region', 'district', 'longitude', 'latitude'],
    color=alt.condition(click, 'region:O', alt.value('lightgray'))
).properties(
    width=500,
    height=500,
)

hf_region = hfacilities2.groupby("region").count().reset_index()
hf_new = pd.DataFrame({'percentage_of_total': (hf_region['facility_name'] / hf_region['facility_name'].sum())})
hf_region = hf_region.join(hf_new)
del hf_new

percentageFacilities = alt.Chart(hf_region).mark_bar().encode(
    y=alt.Y('region:N',  sort=alt.EncodingSortField(op='sum', field='percentage_of_total', order='descending')),
    x=alt.X('percentage_of_total:Q', title='Percentage Of Total', axis=alt.Axis(format='.0%'))
)


c = bar & percentageFacilities | ghana + facilities 
render(c)
map_df.head(2)
mapp = folium.Map(location=[8, 1], zoom_start=6.5, tiles="cartodbpositron")
mapp.choropleth(geo_data=map_df,
              data = map_df, 
              columns = ['region', 'population_to_hfacility'],
              key_on = 'feature.properties.region',
              fill_color = 'YlOrRd', 
              fill_opacity = 0.7, 
              line_opacity = 0.2,
              legend_name = 'Population per Health Facility')

# can't render map with NaNs present in lat/long rows. Rows with latitude missing also have longitude missing, so dropping
# either is enough 
hf = hfacilities.dropna(subset=['latitude'])
FastMarkerCluster(data=hf[['latitude', 'longitude']].values.tolist()).add_to(mapp)

mapp
selection = alt.selection_multi(fields=['region'])

region_bars = alt.Chart().mark_bar().encode(
    y='region',
    x=alt.Y('count(district)', title='Number of districts'),
    color=alt.condition(selection, alt.value('steelblue'), alt.value('lightgray')),
    tooltip='count(district)',
).properties(
    selection=selection,
    width=120,
    height=200
)

heat = alt.Chart().mark_bar().encode(
    alt.X('type'),
    alt.Y('district'),
    color='count()',
    tooltip='count(type)',
).transform_filter(
    filter=selection
).properties(
    width=400,
#     height=500
)

c = alt.hconcat(region_bars, heat, data=hfacilities)
render(c)

