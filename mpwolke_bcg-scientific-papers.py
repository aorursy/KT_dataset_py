#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTZeWFZbMSiMxPj5HL-sfLj4KBjeGXBuIJFJm3N23-IjetTWt9c&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/BCG_COVID-19_scientific_papers.csv', encoding='ISO-8859-2')

df.head()
import time

import warnings

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



warnings.filterwarnings('ignore')
# load metadata

t1 = time.time()

df = pd.read_csv('../input/hackathon/BCG_COVID-19_scientific_papers.csv')

t2 = time.time()

print('Elapsed time:', t2-t1)
# define keyword

my_keyword = 'BCG'
def word_finder(i_word, i_text):

    found = (str(i_text).lower()).find(str(i_word).lower()) # avoid case sensitivity

    if found == -1:

        result = 0

    else:

        result = 1

    return result



# partial function for mapping

word_indicator_partial = lambda text: word_finder(my_keyword, text)

# build indicator vector (0/1) of hits

keyword_indicator = np.asarray(list(map(word_indicator_partial, df.Summary)))
# number of hits

print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())
# add index vector as additional column

df['selection'] = keyword_indicator



# select only hits from data frame

df_hits = df[df['selection']==1]
# show results

df_hits
# show all abstracts

n = df_hits.shape[0]

for i in range(0,n):

    print(df_hits.Title.iloc[i],":\n")

    print(df_hits.Summary.iloc[i])

    print('\n')
# make available for download

df_hits.to_csv('hits.csv')
text = " ".join(abst for abst in df_hits.Summary)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,colormap='Set3',

                      background_color="black").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
# filling missing values with NA

df[['Publication Date', 'BCG COVID-19 Hypothesis category', 'Researcher / Institution', 'Summary', 'Link', 'Journal Source', 'Article type', 'Settings']] = df[['Publication Date', 'BCG COVID-19 Hypothesis category', 'Researcher / Institution', 'Summary', 'Link', 'Journal Source', 'Article type', 'Settings']].fillna('NA')
fig = px.bar(df,

             y='Article type',

             x='BCG COVID-19 Hypothesis category',

             orientation='h',

             color='Settings',

             title='BCG Scientific Papers',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.area(df,

            x='BCG COVID-19 Hypothesis category',

            y='Article type',

            template='plotly_dark',

            color_discrete_sequence=['rgb(18, 115, 117)'],

            title='BCG Scientific Papers',

           )



fig.update_yaxes(range=[0,2])

fig.show()
fig = px.bar(df, 

             x='BCG COVID-19 Hypothesis category', y='Journal Source', color_discrete_sequence=['#27F1E7'],

             title='BCG Scientific Papers ', text='Researcher / Institution')

fig.show()
fig = px.histogram(df[df.Settings.notna()],x="Settings",marginal="box",nbins=10)

fig.update_layout(

    title = "BCG Scientific Papers",

    xaxis_title="Settings",

    yaxis_title="Journal Source",

    template='plotly_dark',

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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSoCtl9zFS72H92rPGUK7Wn9ax5j41BwugbKsY38drS932q_Uot&usqp=CAU',width=400,height=400)