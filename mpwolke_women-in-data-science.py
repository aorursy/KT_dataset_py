#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0WNpRevX2b8A237rMQ2VQaXkQSv20nnmGW2lOFJFlwjI43aGO2w&s',width=400,height=400)
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
wids = pd.read_csv("../input/widsdatathon2020/training_v2.csv")

test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")

sub = pd.read_csv('../input/widsdatathon2020/samplesubmission.csv')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQ44CM9UB1kKjsFpFGU-DTRVmlvGugTz3wPRllnbucMXplydui6w&s',width=400,height=400)
wids.head()
wids.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJE4rPsgdkEJgNcZCknrweDTpASeH3LggoTzdqsZd8iUDTaRz_&s',width=400,height=400)
wids.describe()
print("The number of nulls in each column are \n", wids.isna().sum())
sns.countplot(wids["hospital_death"])

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1XBLX-CmEdjlURIK4ovG5Aflo8QKS_hnycP5kDwlw50C5QRWx&s',width=400,height=400)
sns.distplot(wids["hospital_death"])
sns.scatterplot(x='age',y='hospital_death',data=wids)
print ("Skew is:", wids.hospital_death.skew())

plt.hist(wids.hospital_death, color='pink')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://i.pinimg.com/236x/ac/fb/8b/acfb8b6026740e5c50063307c468f12f.jpg',width=400,height=400)
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
sns.boxplot(x="hospital_death", y="patient_id", data=wids)

#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8TMbfKeH67PQNL0ghQrPrwG75wjca1M7g38Bi9Jbj1T3dsh0F&s',width=400,height=400)
#codes from PSVishnu @psvishnu

hospital = [

    'patient_id','hospital_id','hospital_death','encounter_id']
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxDDQDIfj58v2qLXz1NnNM9fqYg1xdM3cGdJ5d3ktKlMvRpALqqw&s',width=400,height=400)
sns.pairplot(data=wids,diag_kind='kde',vars=hospital,hue='hospital_death')

plt.show()
import plotly.offline as py

value_counts = wids['hospital_id'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "Hospital Id"))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVbT0PMefnpoA7dQYRZKmGvxVO3kXqb6MVCgTcy-guauA-xBX2Cw&s',width=400,height=400)
from collections import Counter

import json

from IPython.display import HTML

import altair as alt

from  altair.vega import v5
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-TVKsdRr0lRfeZHuaMMrzA4g7qN1pOsU-hnd3MoedAtQKTC3T&s',width=400,height=400)


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

        "title": "Hospital - Women in Data Science 2020",

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



def wordcloud_create(wids):

    ult = {}

    corpus = wids.icu_type.values.tolist()

    final = defaultdict(int) #Declaring an empty dictionary for count (Saves ram usage)

    for words in corpus:

        for word in words.split():

             final[word]+=1

    temp = Counter(final)

    for k, v in  temp.most_common(200):

        ult[k] = v

    corpus = pd.Series(ult) #Creating a dataframe from the final default dict

    return render(word_cloud(corpus.to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))
wordcloud_create(wids)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://www.kdnuggets.com/wp-content/uploads/career-progression.jpg',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in wids.apache_2_bodysystem)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhs0JIJjvon1mlAC_QGIIIZDTwGUPY5ZByFTdFaSE9f3l2RC3L2g&s',width=400,height=400)