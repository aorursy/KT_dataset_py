import numpy as np
import pandas as pd
import altair as alt
alt.renderers.enable('notebook')
df=pd.read_csv("../input/adult-arrests-by-county-beginning-1970.csv")
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
df.isna().sum()
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

chart=alt.Chart(df).mark_line().encode(
alt.Y('Total', axis=alt.Axis(title='Total number of Adult Felony'))
,x='Year:N'
,color=alt.Color('County', legend=None),
tooltip=['County','Total']).properties(width=650)
render(chart, id='vega-chart')
brush = alt.selection(type='interval', encodings=['x'])

upper = alt.Chart().mark_line().encode(
    alt.X('Year:N', scale={'domain': brush.ref()}),
    y='Drug Felony:Q'
,color=alt.Color('County', legend=None)
,tooltip=['County','Drug Felony']
).properties(
    width=650,
    height=300
)
lower = upper.properties(
    height=150
).add_selection(
    brush
)
render(alt.vconcat(upper, lower, data=df))
highlight = alt.selection(type='single', on='mouseover',
                          fields=['County'], nearest=True)
base = alt.Chart(df).encode(
    x='Year:N',
    y='Drug Misd:Q',
  color=alt.Color('County:N',legend=None),
 tooltip=['County','Drug Misd'])
points = base.mark_circle().encode(
    opacity=alt.value(0)
).add_selection(
    highlight
).properties(
    width=650
)
lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3))
)

render(points + lines)