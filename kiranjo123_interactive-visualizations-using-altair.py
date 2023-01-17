# Importing required libraries

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns



import altair as alt

alt.data_transformers.enable('default', max_rows=None)



pd.set_option('display.max_columns', 30)

# pd.options.display.max_rows = 1050
import json  # need it for json.dumps

from IPython.display import HTML



# Create the correct URLs for require.js to find the Javascript libraries

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + alt.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



altair_paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {paths}

}});

"""



# Define the function for rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        """Render an altair chart directly via javascript.

        

        This is a workaround for functioning export to HTML.

        (It probably messes up other ways to export.) It will

        cache and autoincrement the ID suffixed with a

        number (e.g. vega-chart-1) so you don't have to deal

        with that.

        """

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

    # Cache will stay defined and keep track of the unique div Ids

    return wrapped





@add_autoincrement

def render_alt(chart, id="vega-chart"):

    # This below is the javascript to make the chart directly using vegaEmbed

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vegaEmbed) {{

        const spec = {chart};     

        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

    }});

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

    workaround.format(paths=json.dumps(altair_paths)),

    "</script>"

)))
df1 = pd.read_csv("../input/Cars93.csv")

df1 = df1.drop(['Unnamed: 0'], axis = 1)
df1.rename(columns={'MPG.city':'MPGC'}, inplace=True)

df1.rename(columns={'MPG.highway':'MPGH'}, inplace=True)

df1.rename(columns={'Luggage.room':'Luggroom'}, inplace=True)

df1.rename(columns={'Fuel.tank.capacity':'Fuelcapacity'}, inplace=True)

df1.rename(columns={'Man.trans.avail':'Gear'}, inplace=True)

df1.rename(columns={'Rev.per.mile':'RevMile'}, inplace=True)

df1.rename(columns={'Turn.circle':'Turn.circle'}, inplace=True)

df1.rename(columns={'Rear.seat.room':'RearseatRoom'}, inplace=True)

df1.head()
interval = alt.selection_interval()



points = alt.Chart(df1).mark_point().encode(

  x='Horsepower',

  y='EngineSize',

  color=alt.condition(interval, 'Origin', alt.value('lightgray'))

).properties(

  selection=interval

)



histogram = alt.Chart(df1).mark_bar().encode(

  x='count()',

  y='Type',

  color='Type'

).transform_filter(interval)



render_alt(points & histogram)
interval = alt.selection_interval()



base = alt.Chart(df1).mark_point().encode(

  y='EngineSize',

  color=alt.condition(interval, 'Origin', alt.value('lightgray'))

).properties(

  selection=interval

)



render_alt(base.encode(x='Weight') | base.encode(x='MPGC'))
points = alt.Chart(df1).mark_point().encode(

  x='Manufacturer',

  y='Price',

  color='Type'

).properties(

  width=800

)



lines = alt.Chart(df1).mark_line().encode(

  x='Manufacturer',

  y='mean(Price)',



).properties(

  width=800

).interactive()

              

render_alt(points + lines)
chart6 = alt.Chart(df1).mark_bar().encode(

    x='Manufacturer',

    y='count()',

    color="Type",

)

render_alt(chart6)
chart4 = alt.Chart(df1).mark_circle().encode(

    alt.X('Horsepower', scale=alt.Scale(zero=False)),

    alt.Y('MPGH', scale=alt.Scale(zero=False, padding=1)),

    color='EngineSize',

    size='Weight'

).interactive()

render_alt(chart4)
chart5 = alt.Chart(df1).mark_line().encode(

    x='Manufacturer',

    y=alt.Y('mean(Horsepower)')

).interactive()

render_alt(chart5)
interval = alt.selection_interval()



base = alt.Chart(df1).mark_point().encode(

  y='Passengers',

  color=alt.condition(interval, 'Type', alt.value('lightgray'))

).properties(

  selection=interval

)



render_alt(base.encode(x='AirBags') | base.encode(x='Manufacturer'))
import seaborn as sns

sns.set(style="whitegrid")

ax = sns.boxplot(x=df1["MPGC"])
chart3 = alt.Chart(df1).mark_bar().encode(

    x='Gear',

    y='count()',

    color='Gear',

    column='Manufacturer'

)



render_alt(chart3)
chart2 = alt.Chart(df1).mark_area(

    opacity=0.3,

    interpolate='step'

).encode(

    alt.X('MPGC', bin=alt.Bin(maxbins=100)),

    alt.Y('count()', stack=None),

    alt.Color(

        'Type'

    )

).interactive()



render_alt(chart2)
chart1 = alt.Chart(df1).mark_rect().encode(

    alt.X('Width', bin=True),

    alt.Y('Fuelcapacity', bin=True),

    alt.Color('count()',

        scale=alt.Scale(scheme='greenblue'),

        legend=alt.Legend(title='Total Records')

    )

).interactive()



render_alt(chart1)