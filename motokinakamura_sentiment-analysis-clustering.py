import pandas as pd

import matplotlib

import numpy as np

import datetime



import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly import tools
df = pd.read_csv('../input/trump-tweets/trumptweets.csv')
df.date = pd.to_datetime(df.date)



df["year"]  = df.date.map(lambda x : x.year)

df["quarter"]  = df.date.map(lambda x : x.quarter)

df["month"]  = df.date.map(lambda x : x.month)

df["day"]  = df.date.map(lambda x : x.day)

df["day_of_week"]  = df.date.map(lambda x : x.strftime('%A'))
!pip install textblob
from textblob import TextBlob
#test



tweet = df.content[1]

tweet
analysis =  TextBlob(tweet)
analysis.sentiment
def sentiment_analysis(text):

    analysis = TextBlob(text)

    Sentiment = analysis.sentiment[0]

    return Sentiment

 



def subjectivity_analysis(text):

    analysis = TextBlob(text)

    Subjectivity = analysis.sentiment[1]

    return Subjectivity 
df["Sentiment"] = df.content.map(lambda x:sentiment_analysis(x) )

df["Subjectivity"] = df.content.map(lambda x:subjectivity_analysis(x) )



df[["Sentiment","Subjectivity"]].head()
px.scatter(df.sample(n=2000), x= "Sentiment", y = "Subjectivity")
df_2020 = df[df.year == 2020]

df_2019 = df[df.year == 2019]

df_2018 = df[df.year == 2018]

df_2017 = df[df.year == 2017]



df_2020_sample = df_2020.sample(n=100)

df_2019_sample = df_2019.sample(n=100)

df_2018_sample = df_2018.sample(n=100)

df_2017_sample = df_2017.sample(n=100)



sentiment_2020 = df_2020_sample.Sentiment

subjectivity_2020 = df_2020_sample.Subjectivity



sentiment_2019 = df_2019_sample.Sentiment

subjectivity_2019 = df_2019_sample.Subjectivity



sentiment_2018 = df_2018_sample.Sentiment

subjectivity_2018 = df_2018_sample.Subjectivity



sentiment_2017 = df_2017_sample.Sentiment

subjectivity_2017 = df_2017_sample.Subjectivity
fig_show_2 = tools.make_subplots(rows=1, cols=4, subplot_titles=("2020","2019","2018","2017"))



scatter_1 = go.Scatter(

    x=sentiment_2020,

    y=subjectivity_2020,

    name='2020',

    mode='markers',

)



scatter_2 = go.Scatter(

    x=sentiment_2019,

    y=subjectivity_2019,

    name='2019',

    mode='markers',

)





scatter_3 = go.Scatter(

    x=sentiment_2018,

    y=subjectivity_2018,

    name='2018',

    mode='markers')



scatter_4 = go.Scatter(

    x=sentiment_2017,

    y=subjectivity_2017,

    name='2017',

    mode='markers',

)



fig_show_2.append_trace(scatter_1, 1, 1)

fig_show_2.append_trace(scatter_2, 1, 2)

fig_show_2.append_trace(scatter_3, 1, 3)

fig_show_2.append_trace(scatter_4, 1, 4)



fig_show_2['layout'].update(height=700, width=900, title = "各年のtweetでランダムに100件を抽出",

                            #yaxis=dict(range=[0, 10]

                           )



fig_show_2
df_kmeans = df[["Sentiment","Subjectivity"]]

df_kmeans.head()
from sklearn.cluster import KMeans
trump_array = np.array(df_kmeans)



n_cluster = 4

pred = KMeans(n_clusters=n_cluster).fit_predict(trump_array)
df_kmeans["pred"] = pred
px.scatter(df_kmeans, x="Sentiment", y="Subjectivity",color="pred")