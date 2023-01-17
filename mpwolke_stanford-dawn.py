#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQR5W8jjiHPaAfVVP7PZRXbXHTJwm1-v51mIMkd22hNOvUspRN9&s',width=400,height=400)
df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Compute Economics/STANFORD DAWN/SQuAD.xlsx')

df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpVh2AelcWmt00NetdTt2678-e-60rfqApsWjntAbMpA19e-GwTg&s',width=400,height=400)
df.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
df.loc[df["Model"] == 0.0, "Hardware"].hist(alpha = 0.5);

df.loc[df["Model"] == 1.0, "Hardware"].hist(alpha = 0.5);
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
df.groupby(['Hardware']).size().plot.bar()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSqb4zoqKpssaJuA8ryb6nxDqSX2a5YkML2m22OESM9lce8V-Mx&s',width=400,height=400)
df.groupby(['Framework']).size().plot.bar()
df.groupby(['Model']).size().plot.bar()
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

value_counts = df['Model'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "Model"))
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

    corpus = df.Model.values.tolist()

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
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='cool')

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Hardware)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQilMJLmma850eij_xGj4f6fJylCbSGLrhyhQq7uV-LZM4ikwLvgA&s',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Framework)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, colormap='Set3', background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Model)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://dawn.cs.stanford.edu/assets/retreat.jpg',width=400,height=400)