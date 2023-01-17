#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTpnmOck-xZlKTvUcwp4rywVsgr34amgR_3AVCyLU3w7wRT7I-A',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import cv2

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://www.mobihealthnews.com/sites/default/files/SPHCC%20and%20Yitu%20Healthcare%20AI%20software_Mobi.jpg',width=400,height=400)
with open('/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/07e833d0917cace550853f72923856d0fe1a7120.json', 'r') as f:

    test = json.load(f)

test.keys()
def affiliation_parsing(x: dict) -> str:

    """Parse affiliation into string."""

    current = []

    for key in ['laboratory', 'institution']:

        if x['affiliation'].get(key):  # could also use try, except

            current.append(x['affiliation'][key])

        else:

            current.append('')

    for key in ['addrLine', 'settlement', 'region', 'country', 'postCode']:

        if x['affiliation'].get('location'):

            if x['affiliation']['location'].get(key):

                current.append(x['affiliation']['location'][key])

        else:

            current.append('')

    return ', '.join(current)



extract_key = lambda x, key: [[i[key] for i in x]]

extract_func = lambda x, func: [[func(i) for i in x]]

format_authors = lambda x: f"{x['first']} {x['last']}"

format_full_authors = lambda x: f"{x['first']} {''.join(x['middle'])} {x['last']} {x['suffix']}"

format_abstract = lambda x: "{}\n{}".format(x['section'], x['text'])

all_keys = lambda x, key: [[i[key] for i in x.values()]]
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://pbs.twimg.com/media/ESnTl26XkAY_RCj?format=jpg&name=small',width=400,height=400)
for path in ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'pmc_custom_license']:

    json_files = [file for file in os.listdir(f'/kaggle/input/CORD-19-research-challenge/2020-03-13/{path}/{path}') if file.endswith('.json')]

    df_list = []



    for js in json_files:

        with open(os.path.join(f'/kaggle/input/CORD-19-research-challenge/2020-03-13/{path}/{path}', js)) as json_file:

            paper = json.load(json_file)

        paper_df = pd.DataFrame({

            'paper_id': paper['paper_id'],

            'title': paper['metadata']['title'],

            'authors': extract_func(paper['metadata']['authors'], format_authors),

            'full_authors': extract_func(paper['metadata']['authors'], format_full_authors),

            'affiliations': extract_func(paper['metadata']['authors'], affiliation_parsing),

            'emails': extract_key(paper['metadata']['authors'], 'email'),

            'raw_authors': [paper['metadata']['authors']],

            'abstract': extract_func(paper['abstract'], format_abstract),

            'abstract_cite_spans': extract_key(paper['abstract'], 'cite_spans'),

            'abstract_ref_spans': extract_key(paper['abstract'], 'ref_spans'),

            'body': extract_func(paper['body_text'], format_abstract),

            'body_cite_spans': extract_key(paper['body_text'], 'cite_spans'),

            'body_ref_spans': extract_key(paper['body_text'], 'ref_spans'),

            'bib_titles': all_keys(paper['bib_entries'], 'title'),

            'raw_bib_entries': [paper['bib_entries']],

            'ref_captions': all_keys(paper['ref_entries'], 'text'),

            'raw_ref_entries': [paper['ref_entries']],

            'back_matter': [paper['back_matter']]

        })

        df_list.append(paper_df)

    temp_df = pd.concat(df_list)

    temp_df.to_csv(f'/kaggle/working/{path}.csv', index=False)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://healthmanagement.org/uploads/from_cloud/cw/00116374_cw_image_wi_4b727520e6ffc2fb5088f4be9576f7b9.jpg.pagespeed.ce.oZjkUhglUi.jpg',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS6kGQzHbJKLbb5ZLG1ezKfpmUlcfUFAlp0O-ARn9cYcyxAiN3F',width=400,height=400)
df_list = []

for path in ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'pmc_custom_license']:

    temp_df = pd.read_csv(f'/kaggle/working/{path}.csv')

    temp_df['dataset'] = path

    df_list.append(temp_df)

    

aggregate_df = pd.concat(df_list)

aggregate_df.to_csv(f'/kaggle/working/all_df.csv', index=False)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRGKG7NakimAmL2UNPbjwMiR6c0xBzMkgg4jnrG1WmCrBt4YdhZ',width=400,height=400)
aggregate_df  # view the aggregated data
aggregate_df.dtypes
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

    corpus = aggregate_df.affiliations.values.tolist()

    final = defaultdict(int) #Declaring an empty dictionary for count (Saves ram usage)

    for words in corpus:

        for word in words.split():

             final[word]+=1

    temp = Counter(final)

    for k, v in  temp.most_common(200):

        ult[k] = v

    corpus = pd.Series(ult) #Creating a dataframe from the final default dict

    return render(word_cloud(corpus.to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))
wordcloud_create(aggregate_df)
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/000_1ov3n5_0.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Create our shapening kernel, we don't normalize since the 

# the values in the matrix sum to 1

kernel_sharpening = np.array([[-1,-1,-1], 

                              [-1,9,-1], 

                              [-1,-1,-1]])



# applying different kernels to the input image

sharpened = cv2.filter2D(image, -1, kernel_sharpening)





plt.subplot(1, 2, 2)

plt.title("Image Sharpening")

plt.imshow(sharpened)



plt.show()
# Load our new image

image = cv2.imread('/kaggle/input//medical-masks-dataset/images/003_1024.jpeg', 0)



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



# Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



plt.subplot(3, 2, 2)

plt.title("Threshold Binary")

plt.imshow(thresh1)



# It's good practice to blur images as it removes noise

image = cv2.GaussianBlur(image, (3, 3), 0)



# Using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.title("Otsu's Thresholding")

plt.imshow(th2)





plt.subplot(3, 2, 5)

# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3)

plt.show()
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/002_1024.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



# Let's define our kernel size

kernel = np.ones((5,5), np.uint8)



# Now we erode

erosion = cv2.erode(image, kernel, iterations = 1)



plt.subplot(3, 2, 2)

plt.title("Erosion")

plt.imshow(erosion)



# 

dilation = cv2.dilate(image, kernel, iterations = 1)

plt.subplot(3, 2, 3)

plt.title("Dilation")

plt.imshow(dilation)





# Opening - Good for removing noise

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.subplot(3, 2, 4)

plt.title("Opening")

plt.imshow(opening)



# Closing - Good for removing noise

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.subplot(3, 2, 5)

plt.title("Closing")

plt.imshow(closing)
# Let's load a simple image with 3 black squares

image = cv2.imread('/kaggle/input/medical-masks-dataset/images/so(19).jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)





# Grayscale

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



# Find Canny edges

edged = cv2.Canny(gray, 30, 200)



plt.subplot(2, 2, 2)

plt.title("Canny Edges")

plt.imshow(edged)



# Finding Contours

# Use a copy of your image e.g. edged.copy(), since findContours alters the image

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



plt.subplot(2, 2, 3)

plt.title("Canny Edges After Contouring")

plt.imshow(edged)



print("Number of Contours found = " + str(len(contours)))



# Draw all contours

# Use '-1' as the 3rd parameter to draw all

cv2.drawContours(image, contours, -1, (0,255,0), 3)



plt.subplot(2, 2, 4)

plt.title("Contours")

plt.imshow(image)
import numpy as np

import pandas as pd 

import cv2

from fastai.vision import *

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import os

import shutil

from glob import glob

%matplotlib inline

!pip freeze > '../working/dockerimage_snapshot.txt'
def makeWordCloud(df,column,numWords):

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def plotImages(artist,directory):

    print(artist)

    multipleImages = glob(directory)

    plt.rcParams['figure.figsize'] = (15, 15)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    for l in multipleImages[:25]:

        im = cv2.imread(l)

        im = cv2.resize(im, (128, 128)) 

        plt.subplot(5, 5, i_+1) #.set_title(l)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        i_ += 1



np.random.seed(7)
print(os.listdir("../input/medical-masks-dataset/images/"))
img_dir='../input/medical-masks-dataset/images'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),

                                  size=299,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))
cnt_srs = aggregate_df['dataset'].value_counts().head()

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

    title='Dataset distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="dataset")
cnt_srs = aggregate_df['abstract'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Abstracts distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="abstract")
fig = px.pie( values=aggregate_df.groupby(['dataset']).size().values,names=aggregate_df.groupby(['dataset']).size().index)

fig.update_layout(

    title = "dataset",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(aggregate_df[aggregate_df.dataset.notna()],x="dataset",marginal="box",nbins=10)

fig.update_layout(

    title = "dataset",

    xaxis_title="dataset",

    yaxis_title="Number of datasets",

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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ2WW7WwEFT0lnnLSu9j4XAGpgW7N-gYJODifJ02D0j6Q3z2MrK',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT72jGRpDdXhCyNI7k28qbxadRkMeKMMC0-5wZjblpDj35loLuX',width=400,height=400)