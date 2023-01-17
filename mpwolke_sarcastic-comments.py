# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

# train-balanced-sarcasm.csv has 1010826 rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('../input/sarcastic-comments-on-reddit/train-balanced-sarcasm.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'train-balanced-sarcasm.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.dtypes


#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3xjPuH9PdZzMlj1VpEKlj7VVE-RkAmIjN4Jc7jaWmwMwFE_N4dg&s',width=400,height=400)
corrs = df.corr()

corrs
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('SARCASM');
sns.distplot(df["ups"])
sns.distplot(df["downs"])
sns.scatterplot(x='score',y='downs',data=df)
sns.countplot(df["ups"])

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
labels1=df.score.value_counts().index

sizes1=df.score.value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")

plt.title("score",size=25)

plt.show()
print ("Skew is:", df.score.skew())

plt.hist(df.score, color='pink')

plt.show()
import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['score', 'downs'], as_index=False).ups.sum()



fig = px.bar(plot_data, x='score', y='ups', color='downs')

fig.update_layout(

    title_text='Sarcastic comments',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['score', 'downs'], as_index=False).ups.sum()



fig = px.line_polar(plot_data, theta='score', r='ups', color='downs')

fig.update_layout(

    title_text='Sarcastic Comments',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Genre and artist

plot_data = df.groupby(['score', 'downs'], as_index=False).ups.sum()



fig = px.line(plot_data, x='score', y='ups', color='downs')

fig.update_layout(

    title_text='Sarcastic Comments',

    height=500, width=1000)

fig.show()
#sample codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

fig = go.Figure(go.Treemap(

    labels = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],

    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"]

))



fig.show()
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#make a df it's grouped by "Genre"

gb_score =df.groupby("score").sum()



gb_score.head()
# codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

score = list(gb_score.index)

sarc = list(gb_score.ups)



print(score)

print(sarc)
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#first treemap

test_tree = go.Figure(go.Treemap(

    labels =  score,

    parents=[""]*len(score),

    values =  sarc,

    textinfo = "label+value"

))



test_tree.show()
#codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

#second treemap

test_tree_blue = go.Figure(go.Treemap(

    labels =  score,

    parents=[""]*len(score),

    values =  sarc,

    textinfo = "label+value",

    marker_colorscale = 'magma'

))



test_tree_blue.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2Ymvgxnf4XsEy1UnrjJM4-v4cQISc20XrI-q3gucZD61WmN_0&s',width=400,height=400)
# Necessary Functions: 

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
import plotly.offline as py

value_counts = df['score'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "Type Distribution"))
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

    corpus = df.comment.values.tolist()

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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTllzoOJzbtATW5rkE6l5uPVALudyrMLrr--zkLCb9M3pSiOuOG&s',width=400,height=400)