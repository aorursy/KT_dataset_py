!pip install discover_feature_relationships

!pip install hvplot
import pandas as pd

import numpy as np

import seaborn as sns

from discover_feature_relationships import discover

import sklearn

import holoviews as hv

import geoviews as gv

import geopandas as gpd

import hvplot.pandas

from cartopy import crs

import pycountry

from fuzzywuzzy import fuzz



import altair as alt

import matplotlib.pyplot as plt

%matplotlib inline

alt.renderers.enable('notebook')

hv.extension('bokeh', 'matplotlib')

gv.extension('bokeh', 'matplotlib')
from IPython.display import HTML



import altair as alt

from  altair.vega import v3

import json



vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

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

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))
df_2015 = pd.read_csv("../input/2015.csv")

df_2016 = pd.read_csv("../input/2016.csv")

df_2017 = pd.read_csv("../input/2017.csv")
targets = ['Low', 'Low-Mid', 'Top-Mid', 'Top']

h_cols = ['Country', 'GDP', 'Family', 'Life', 'Freedom', 'Generosity', 'Trust']

def prep_frame(df_year, year):

    df = pd.DataFrame()

    # Work around to load 2015, 2016, 2017 data into one common column

    target_cols = []

    for c in h_cols:

        target_cols.extend([x for x in df_year.columns if c in x])

    df[h_cols] = df_year[target_cols]

    df['Happiness Score'] = df_year[[x for x in df_year.columns if 'Score' in x]]

    # Calculate quartiles on the data.

    df["target"] = pd.qcut(df[df.columns[-1]], len(targets), labels=targets)

    df["target_n"] = pd.qcut(df[df.columns[-2]], len(targets), labels=range(len(targets)))

    # Append year and assign to multi-index

    df['Year'] = year

    df = df.set_index(['Country', 'Year'])

    return df

df = prep_frame(df_2015, 2015)

df = df.append(prep_frame(df_2016, 2016), sort=False)

df = df.append(prep_frame(df_2017, 2017), sort=False)

df.head()
spearman_cormatrix= df.corr(method='spearman')

spearman_cormatrix
fig, ax = plt.subplots(ncols=2,figsize=(24, 8))

sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1, ax=ax[0], center=0, cmap="viridis", annot=True)

sns.heatmap(spearman_cormatrix, vmin=-.25, vmax=1, ax=ax[1], center=0, cmap="Accent", annot=True)
sns.pairplot(df.drop(['target_n'], axis=1), hue='target')

#hvplot.scatter_matrix(df.drop(['target_n'], axis=1), c='target')



#plt.show()
classifier_overrides = set()

df_results = discover.discover(df.drop(['target', 'target_n'],axis=1).sample(frac=1), classifier_overrides)
fig, ax = plt.subplots(ncols=2,figsize=(24, 8))

sns.heatmap(df_results.pivot(index='target', columns='feature', values='score').fillna(1).loc[df.drop(['target', 'target_n'],axis=1).columns,df.drop(['target', 'target_n'],axis=1).columns],

            annot=True, center=0, ax=ax[0], vmin=-1, vmax=1, cmap="viridis")

sns.heatmap(df_results.pivot(index='target', columns='feature', values='score').fillna(1).loc[df.drop(['target', 'target_n'],axis=1).columns,df.drop(['target', 'target_n'],axis=1).columns],

            annot=True, center=0, ax=ax[1], vmin=-0.25, vmax=1, cmap="Accent")

plt.plot()
#from sklearn.decomposition import PCA

from sklearn.decomposition import MiniBatchSparsePCA as PCA

pca = PCA(n_components=2,

          batch_size=10,

          normalize_components=True,

          random_state=42)

principalComponents = pca.fit_transform(df[h_cols[1:-2]])



source = df.copy()

source['component 1'] = principalComponents[:,0]

source['component 2'] = principalComponents[:,1]

source.head()
base = alt.Chart(source.reset_index())



xscale = alt.Scale(domain=(source['component 1'].min(), source['component 1'].max()))

yscale = alt.Scale(domain=(source['component 2'].min(), source['component 2'].max()))



area_args = {'opacity': .6, 'interpolate': 'step'}



points = base.mark_circle(size=60).encode(

    alt.X('component 1', scale=xscale),

    alt.Y('component 2', scale=yscale),

    color='target',

    tooltip=['Country', 'target', 'GDP', 'Family', 'Life']

).properties(height=600,width=600).interactive()





top_hist = base.mark_area(**area_args).encode(

    alt.X('component 1:Q',

          # when using bins, the axis scale is set through

          # the bin extent, so we do not specify the scale here

          # (which would be ignored anyway)

          bin=alt.Bin(maxbins=20, extent=xscale.domain),

          stack=None,

          title=''

         ),

    alt.Y('count()', stack=None, title=''),

    alt.Color('target:N'),

).properties(height=60,width=600)



right_hist = base.mark_area(**area_args).encode(

    alt.Y('component 2:Q',

          bin=alt.Bin(maxbins=20, extent=yscale.domain),

          stack=None,

          title='',

         ),

    alt.X('count()', stack=None, title=''),

    alt.Color('target:N'),

).properties(width=60,height=600)



render(top_hist & (points | right_hist))



from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

tmp_df = df.iloc[df.index.get_level_values('Year') == 2017].reset_index()

tmp_df.loc[:,["Happiness Score"]+h_cols[1:]] = min_max_scaler.fit_transform(tmp_df[["Happiness Score"]+h_cols[1:]])
hvplot.parallel_coordinates(tmp_df, 'target', cols=["Happiness Score"]+h_cols[1:], alpha=.3, tools=['hover', 'tap'], width=800, height=500)
tmp_df.sort_values(by='Generosity', ascending=False).head()
rank_df = tmp_df[h_cols[:4]].rank(axis=0,numeric_only=True, method='dense', ascending=False)

rank_df['Country'] = tmp_df['Country']

rank_df['Influence'] = tmp_df[h_cols].rank(axis=0,numeric_only=True, method='dense').idxmax(axis=1)

rank_df['True Influence'] = tmp_df[h_cols[:4]].rank(axis=0,numeric_only=True, method='dense').idxmax(axis=1)
# Country names are hard.

countries = {}

for country in pycountry.countries:

    countries[country.alpha_3] = country.name

world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world_map['Country'] = [countries.get(country, 'Unknown Code') for country in list(world_map['iso_a3'])]



for q in world_map['Country']:

    if "Unknown Code" in q:

        world_map.loc[world_map.Country == q, 'Country'] = world_map.loc[world_map.Country == q, 'name']

    elif q in "Ivory Coast":

        world_map.loc[world_map.Country == q, 'Country'] = "Côte d'Ivoire"

    elif q in "Viet Nam":

        world_map.loc[world_map.Country == q, 'Country'] = "Vietnam"

    elif "Korea" in q:

        world_map.loc[world_map.Country == q, 'Country'] = "South Korea"

        



for x in rank_df['Country']:

    if not x in list(world_map['Country']):

        for q in world_map['Country']:

            if (x[:5] in q) and (not x[:5] in "South"):

                world_map.loc[world_map.Country == q, 'Country'] = x

                break

            elif fuzz.partial_ratio(x,q) > 75:

                world_map.loc[world_map.Country == q, 'Country'] = x

                break

        else:

            if not x in list(world_map['name']):

                world_map.loc[world_map.Country == q, 'Country'] = x

            



with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(world_map[['iso_a3', 'name', 'Country']])
gv_frame = pd.merge(world_map, rank_df, on='Country')



background = gv.Polygons(gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))).opts(color="#FFFFFF")

clusters = gv.Polygons(gv_frame, vdims=['True Influence', 'Influence', 'Country']).opts(tools=['hover', 'tap'], cmap='Accent', show_legend=True, legend_position='bottom_left')



((background * clusters).opts(width=800, height=500, projection=crs.PlateCarree()))

background = gv.Polygons(gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))).opts(color="#FFFFFF")

clusters = gv.Polygons(gv_frame, vdims=['Influence', 'True Influence' ,'Country']).opts(tools=['hover', 'tap'], cmap='Dark2', show_legend=True, legend_position='bottom_left')



((background * clusters).opts(width=800, height=500, projection=crs.PlateCarree()))