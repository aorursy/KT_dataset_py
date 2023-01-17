import numpy as np
import pandas as pd
import altair as alt
alt.renderers.enable('notebook')
df=pd.read_csv("../input/nyjob1.csv")
df.head(5) # Reading first few observations
# Setup
import csv
import json
import re
import numpy as np
import pandas as pd
import altair as alt

from collections import Counter, OrderedDict
from IPython.display import HTML
from  altair.vega import v3

# The below is great for working but if you publish it, no charts show up.
# The workaround in the next cell deals with this.
#alt.renderers.enable('notebook')

HTML("This code block contains import statements and setup.")

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
df.columns
df.columns=['Area', 'Year', 'Month', 'Labor Force', 'Employed', 'Unemployed',
       'UnemploymentRate']
df['UnemploymentRate']=df['UnemploymentRate'].str.replace('%',"").astype(float)
df.head(5) # First 5 observations without % symbol in "UnemploymnetRate" column
df1=df[(df.Area=='New York City') | (df.Area=='New York State')| (df.Area=='New York City Region')]
df1.head(5) #Subsetted dataframe contains Area as NY only
plot1=alt.Chart(df1).mark_area().encode(
x='Year:N'
,y=alt.Y('mean(UnemploymentRate):Q',title=None)
,color='Area:N'
,row='Area:N').properties(width=500,height=140)

render(plot1)
df2=df[(df.Area=='Bronx County')|(df.Area=='Kings County')| (df.Area=='Queens County')]
plot2=alt.Chart(df2).mark_circle(opacity=1).encode(
    x="Year:N",
    y="UnemploymentRate:Q",
    color="Area:N"
)
render(plot2)
multi=alt.selection_multi(encodings=['color'])

plot3=alt.Chart(df2).mark_circle().encode(
    x="Year:N",
    y="UnemploymentRate:Q",
     color=alt.condition(multi, 'Area', alt.value('lightgray'))
).properties(
    selection=multi
)
render(plot3)
click=alt.selection_multi(encodings=['color'])
scatter=alt.Chart(df2).mark_circle().encode(
    x="Year:N",
    y="UnemploymentRate:Q",
     color=alt.Color('Area:N',legend=None)
).transform_filter(
    click
)
pn= alt.Chart(df2).mark_circle().encode(
    y="Area:O",
     color=alt.condition(click, 'Area', alt.value('lightgray'),legend=None)
).properties(
    selection=click
)
render(scatter | pn)
plot4=alt.Chart(df2).mark_rect().encode(
    x='Year:N',
    y='Area:N',
    color=alt.Color('UnemploymentRate:Q',scale=alt.Scale(scheme='orangered'),legend=None)
)
render(plot4)
#Unemployment Rate in scale between 4-14%