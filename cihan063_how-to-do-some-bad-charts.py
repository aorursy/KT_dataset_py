# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
us_videos=pd.read_csv('../input/youtube-new/USvideos.csv')

gb_videos = pd.read_csv('../input/youtube-new/GBvideos.csv')
gb_videos.head()
us_videos= us_videos.drop(['description','tags','thumbnail_link','video_id','comments_disabled','ratings_disabled','video_error_or_removed'],axis=1)

gb_videos= gb_videos.drop(['description','tags','thumbnail_link','video_id','comments_disabled','ratings_disabled','video_error_or_removed'],axis=1)
us_videos.head()
gb_videos.head()
df1=pd.DataFrame(us_videos)

df2=pd.DataFrame(gb_videos)
us_videos['country']='US'

gb_videos['country']='GB'

result_df=us_videos.append([gb_videos])

shuffled_df=result_df.sample(frac=1).reset_index()

shuffled_df.head(15)
import plotly.express as px
fig = px.pie(us_videos, values='views', names='category_id', title='Most watched categories in US',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
fig = px.pie(gb_videos, values='views', names='category_id', title='Most watched categories in GB',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
fig = px.scatter(shuffled_df, x="likes", y="dislikes", facet_col="country",

                 width=800, height=400)

fig.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),

    paper_bgcolor="LightSteelBlue",

)



fig.show()
shuffled_100=shuffled_df.iloc[:100,:]

shuffled_100.head()
from wordcloud import WordCloud
x_channelTitle=shuffled_100.channel_title[shuffled_100.country != ''] #I want to use both of GB and US. Therefore if country not null. It use both of them.

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x_channelTitle))

plt.imshow(wordcloud)

plt.axis('off')



plt.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots
fig = px.scatter_3d(shuffled_df, x='likes', y='dislikes', z='comment_count',color='category_id',symbol='category_id')

fig.show()
df = px.data.gapminder()

fig = px.scatter_3d(shuffled_df, x='category_id', y='country', z='likes', size='index', color='category_id')

fig.update_layout(scene_zaxis_type="log")

fig.show()
from collections import Counter
categories=Counter(shuffled_df['category_id'])

most_common_categories=categories.most_common(5)

x,y= zip(*most_common_categories)

x,y= list(x), list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.hls_palette(len(x)))

plt.xlabel('Most Common Categories')
from plotly.offline import iplot
x=shuffled_100.category_id



trace1 = {

  'x': x,

  'y': shuffled_100.likes,

  'name': 'likes',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': shuffled_100.dislikes,

  'name': 'dislikes',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
categories=Counter(shuffled_df['category_id'])

most_common_categories=categories.most_common(5)

x,y= zip(*most_common_categories)

x,y= list(x), list(y)

x1_df=pd.DataFrame(x,y)

x1_df
plt.subplots(figsize=(8,5))

sns.swarmplot(x=shuffled_100['country'],y=shuffled_100['views'],hue=shuffled_100['category_id'])

plt.show()
df = px.data.tips()



fig = px.box(shuffled_100, x="category_id", y="comment_count", color="country")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default

fig.show()
x_Comment=shuffled_100.comment_count[shuffled_100.country != '']

x_Dislikes=shuffled_100.dislikes[shuffled_100.country != '']



trace1 = go.Histogram(

    x=x_Comment,

    opacity=0.75,

    name = "Comments",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=x_Dislikes,

    opacity=0.75,

    name = "Dislikes",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Comments - Dislikes Rate',                   

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = [

    {

        'y': shuffled_100.likes,

        'x': shuffled_100.dislikes,

        'mode': 'markers',

        'marker': {

            'color':shuffled_100.category_id,

            'size':shuffled_100.category_id,

            'showscale': True

        },

        "text" :shuffled_100.category_id    

    }

]

iplot(data)