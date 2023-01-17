import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



### Plotly

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.plotly as py

from plotly import tools

init_notebook_mode(connected=True)





# Altair

import altair as alt



### Removes warnings that occassionally show up

import warnings

warnings.filterwarnings('ignore')
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
train = pd.read_csv("../input/Churn_Modelling.csv")
train.head()
target_col = ["Exited"]

cat_cols   = train.nunique()[train.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

num_cols   = [x for x in train.columns if x not in cat_cols + target_col]

num_cols

from altair import pipe, limit_rows, to_values

t = lambda data: pipe(train, limit_rows(max_rows=10000), to_values)

alt.data_transformers.register('custom', t)

alt.data_transformers.enable('custom')
interval = alt.selection_interval()



points = alt.Chart(train).mark_point().encode(

  x='Age',

  y='Balance',

  color=alt.condition(interval, 'Geography', alt.value('lightgray'))

).properties(

  selection=interval

)



histogram = alt.Chart(train).mark_bar().encode(

  x='count()',

  y='Geography',

  color='Geography'

).transform_filter(interval)



render_alt(points & histogram)
exited_1 = train.query('Exited==1')

exited_0 = train.query('Exited==0')
#function  for histogram for customer churn types

def histogram(column) :

    trace1 = go.Histogram(x  = exited_1[column],

                          histnorm= "percent",

                          name = "Exited",

                          

                          marker = dict(line = dict(width = .5,

                                                    color = "black"

                                                    ), color = '#dd3b3b'

                                        ),

                         opacity = .9 

                         ) 

    

    trace2 = go.Histogram(x  = exited_0[column],

                          histnorm = "percent",

                          name = "Not Exited",

                         

                          marker = dict(line = dict(width = .5,

                                              color = "black"

                                             ), color = '#336dcc'

                                 ),

                          opacity = .9

                         )

    

    data = [trace1,trace2]

    layout = go.Layout(dict(title =column + " Distribution",

                           

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = column,

                                             

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=2

                                            ),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = "Percent",

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=2

                                            ),

                           )

                      )

    fig  = go.Figure(data=data,layout=layout)

    

    iplot(fig)
histogram('Age')
histogram('Balance')
histogram('CreditScore')
trace = []

def gen_boxplot(df):

    for feature in df:

        trace.append(

            go.Box(

                name = feature,

                y = df[feature]

            )

        )



new_df = train[num_cols[6:]]

gen_boxplot(new_df)

data = trace

iplot(data)
brush = alt.selection(type='interval', encodings=['x'])



bars =alt.Chart(train).mark_bar().encode(

    alt.X('Balance', bin=True),

    alt.Y('count()'),

    alt.Color('Geography'),

    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7))

).add_selection(

    brush

)





render_alt(alt.layer(bars,  data=train))
brush = alt.selection(type='interval', encodings=['x'])



bars = alt.Chart().mark_bar().encode(

    x=alt.X('Age'),

    y=alt.Y('mean(EstimatedSalary)',scale=alt.Scale(domain=(40000, 200000))),

    #x='CreditScore',

    #y='mean(EstimatedSalary)',

    #color='Geography',

    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7))

).add_selection(

    brush

)



line = alt.Chart().mark_rule(color='firebrick').encode(

    y='mean(EstimatedSalary)',

    size=alt.SizeValue(3)

).transform_filter(

    brush

)



render_alt(alt.layer(bars, line, data=train))
target = "Geography"

feature = "CreditScore"



fig = ff.create_distplot(

    [train[train[target] == y][feature].values for y in train[target].unique()], 

    train[target].unique(), 

    show_hist=False,

    show_rug=False,

)



for d in fig['data']:

    d.update({'fill' : 'tozeroy'})



layout = go.Layout(

    title   = "Country-wise Credit Behaviour",

    xaxis   = dict(title = "Credit"),

    yaxis   = dict(title = "Density"),

)



fig["layout"] = layout

iplot(fig)