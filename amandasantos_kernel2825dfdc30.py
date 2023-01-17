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



##-----------------------------------------------------------

# This whole section 

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

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))
import numpy as np

import scipy as sp

import pandas as pd



###########code one lab in python

import numpy as np

import scipy as sp

import pandas as pd



import matplotlib as mpl

import matplotlib.pyplot as plt







housing = pd.read_csv('../input/USDA_RD_July_2016_1.csv')
housing.head()
import altair as alt

alt.renderers.enable('notebook')

interval = alt.selection_interval()



points = alt.Chart(housing).mark_point().encode(

  

    x=alt.X('Total_1_Bedroom_Units', axis=alt.Axis( title='One Bedroom units')),

    y=alt.Y('Median_Household_Income_Obligation', axis=alt.Axis( title='Median Household Income')),

  

  color=alt.condition(interval, 'Construction_Type',alt.value('lightgray'))

).properties(

  selection=interval

)



histogram = alt.Chart(housing).mark_bar().encode(

  x='count()',

  y='Construction_Type',

  color='Construction_Type'

).transform_filter(interval)



render(points & histogram) 

brush = alt.selection(type='interval', encodings=['x'])



bars = alt.Chart(housing).mark_bar().encode(

    x=alt.X('Lender_Name', axis=alt.Axis( title='Lender Name')),

    y=alt.Y('mean(Loan_to_Cost):Q', axis=alt.Axis( title='Average of loan to Cost Ratio')),

    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7))

).add_selection(

    brush

)



line = alt.Chart().mark_rule(color='firebrick').encode(

    y='mean(Loan_to_Cost):Q',

    size=alt.SizeValue(3)

).transform_filter(

    brush

)



render(alt.layer(bars, line, data=housing))


