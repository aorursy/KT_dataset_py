# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19-allresearchpapers-lemmatizedinformation/COVID-19_AllResearchPapers_LemmatizedInformation.csv")
df.head()
df.dtypes
def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label+value",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig
from collections import Counter

import json

from IPython.display import HTML

import altair as alt

from  altair.vega import v5
##-----------------------------------------------------------

# This whole section 

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

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

    "</script>")))
def word_cloud(df, pixwidth=6000, pixheight=350, column="index", counts="count"):

    data= [dict(name="dataset", values=df.to_dict(orient="records"))]

    wordcloud = {

        "$schema": "https://vega.github.io/schema/vega/v5.json",

        "width": pixwidth,

        "height": pixheight,

        "padding": 0,

        "title": "Hover to see number of occureances from all the sequences",

        "data": data

    }

    scale = dict(

        name="color",

        type="ordinal",

        range=["cadetblue", "royalblue", "steelblue", "navy", "teal"]

    )

    mark = {

        "type":"text",

        "from":dict(data="dataset"),

        "encode":dict(

            enter=dict(

                text=dict(field=column),

                align=dict(value="center"),

                baseline=dict(value="alphabetic"),

                fill=dict(scale="color", field=column),

                tooltip=dict(signal="datum.count + ' occurrances'")

            )

        ),

            "transform": [{

            "type": "wordcloud",

            "text": dict(field=column),

            "size": [pixwidth, pixheight],

            "font": "Helvetica Neue, Arial",

            "fontSize": dict(field="datum.{}".format(counts)),

            "fontSizeRange": [10, 60],

            "padding": 2

        }]

    }

    wordcloud["scales"] = [scale]

    wordcloud["marks"] = [mark]

    

    return wordcloud



from collections import defaultdict



def wordcloud_create(df):

    ult = {}

    corpus = df.text.values.tolist()

    final = defaultdict(int) #Declaring an empty dictionary for count (Saves ram usage)

    for words in corpus:

        for word in words.split():

             final[word]+=1

    temp = Counter(final)

    for k, v in  temp.most_common(200):

        ult[k] = v

    corpus = pd.Series(ult) #Creating a dataframe from the final default dict

    return render(word_cloud(corpus.to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))
wordcloud_create(df)
cnt_srs = df['document_keyword'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Document Keyword distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="document_keyword")
fig = px.pie( values=df.groupby(['affiliations']).size().values,names=df.groupby(['affiliations']).size().index)

fig.update_layout(

    title = "",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df[df.document_keyword.notna()],x="paper_id",marginal="box",nbins=10)

fig.update_layout(

    title = "Paper ID",

    xaxis_title="paper_id",

    yaxis_title="Number of paper-ids",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)