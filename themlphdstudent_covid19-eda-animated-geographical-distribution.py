import numpy as np

import pandas as pd 

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from iso3166 import countries

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
df.head()
df.info()
missed = pd.DataFrame()

missed['column'] = df.columns



missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]

missed = missed.sort_values('percent')

missed = missed[missed['percent']>0]



fig = px.bar(

    missed, 

    x='percent', 

    y="column", 

    orientation='h', 

    title='Missed values percent for every column (percent > 0)', 

    height=400, 

    width=600

)

fig.show()
ds = df['user_name'].value_counts().reset_index()

ds.columns = ['user_name', 'tweets_count']

ds = ds.sort_values(['tweets_count'])

fig = px.bar(

    ds.tail(40), 

    x="tweets_count", 

    y="user_name", 

    orientation='h', 

    title='Top 40 users by number of tweets', 

    width=800, 

    height=800

)

fig.show()
df = pd.merge(df, ds, on='user_name')
data = df.sort_values('user_followers', ascending=False)

data = data.drop_duplicates(subset='user_name', keep="first")

data = data[['user_name', 'user_followers', 'tweets_count']]

data = data.sort_values('user_followers')

fig = px.bar(

    data.tail(40), 

    x="user_followers", 

    y="user_name", 

    color='tweets_count',

    orientation='h', 

    title='Top 40 users by number of followers', 

    width=800, 

    height=800

)

fig.show()
data = df.sort_values('user_friends', ascending=False)

data = data.drop_duplicates(subset='user_name', keep="first")

data = data[['user_name', 'user_friends', 'tweets_count']]

data = data.sort_values('user_friends')

fig = px.bar(

    data.tail(40), 

    x="user_friends", 

    y="user_name", 

    color = 'tweets_count',

    orientation='h', 

    title='Top 40 users by number of friends', 

    width=800, 

    height=800

)

fig.show()
df['user_created'] = pd.to_datetime(df['user_created'])

df['year_created'] = df['user_created'].dt.year

data = df.drop_duplicates(subset='user_name', keep="first")

data = data[data['year_created']>1970]



data = data['year_created'].value_counts().reset_index()

data.columns = ['year', 'number']



fig = px.bar(

    data, 

    x="year", 

    y="number", 

    orientation='v', 

    title='User created year by year', 

    width=800, 

    height=600

)

fig.show()
df.head(10)
ds = df['user_location'].value_counts().reset_index()

ds.columns = ['user_location', 'count']

ds = ds[ds['user_location']!='NA']

ds = ds.sort_values(['count'])

fig = px.bar(

    ds.tail(40), 

    x="count", 

    y="user_location", 

    orientation='h', title='Top 40 user locations by number of tweets', 

    width=800, 

    height=800

)

fig.show()
def pie_count(data, field, percent_limit, title):

    

    data[field] = data[field].fillna('NA')

    data = data[field].value_counts().to_frame()



    total = data[field].sum()

    data['percentage'] = 100 * data[field]/total    



    percent_limit = percent_limit

    otherdata = data[data['percentage'] < percent_limit] 

    others = otherdata['percentage'].sum()  

    maindata = data[data['percentage'] >= percent_limit]



    data = maindata

    other_label = "Others(<" + str(percent_limit) + "% each)"

    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 

    

    labels = data.index.tolist()   

    datavals = data[field].tolist()

    

    trace=go.Pie(labels=labels,values=datavals)



    layout = go.Layout(

        title = title,

        height=600,

        width=600

        )

    

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)

    

pie_count(df, 'user_location', 0.5, 'Number of tweets per location')
ds = df['source'].value_counts().reset_index()

ds.columns = ['source', 'count']

ds = ds.sort_values(['count'])

fig = px.bar(

    ds.tail(40), 

    x="count", 

    y="source", 

    orientation='h', 

    title='Top 40 user sources by number of tweets', 

    width=800, 

    height=800

)

fig.show()
df['hashtags'] = df['hashtags'].fillna('[]')

df['hashtags_count'] = df['hashtags'].apply(lambda x: len(x.split(',')))

df.loc[df['hashtags'] == '[]', 'hashtags_count'] = 0

df.head(10)
df['hashtags_count'].describe()
ds = df['hashtags_count'].value_counts().reset_index()

ds.columns = ['hashtags_count', 'count']

ds = ds.sort_values(['count'])

ds['hashtags_count'] = ds['hashtags_count'].astype(str) + ' tags'

fig = px.bar(

    ds, 

    x="count", 

    y="hashtags_count", 

    orientation='h', 

    title='Distribution of number of hashtags in tweets', 

    width=800, 

    height=600

)

fig.show()
ds = df[df['tweets_count']>10]

ds = ds.groupby(['user_name', 'tweets_count'])['hashtags_count'].mean().reset_index()

ds.columns = ['user', 'tweets_count', 'mean_count']

ds = ds.sort_values(['mean_count'])

fig = px.bar(

    ds.tail(40), 

    x="mean_count", 

    y="user", 

    color='tweets_count',

    orientation='h', 

    title='Top 40 users with higher mean number of hashtags (at least 10 tweets per user)', 

    width=800, 

    height=800

)

fig.show()
df['date'] = pd.to_datetime(df['date']) 

df = df.sort_values(['date'])

df['day'] = df['date'].astype(str).str.split(' ', expand=True)[0]

df['time'] = df['date'].astype(str).str.split(' ', expand=True)[1]

df.head()
ds = df.groupby(['day', 'user_name'])['hashtags_count'].count().reset_index()

ds = ds.groupby(['day'])['user_name'].count().reset_index()

ds.columns = ['day', 'number_of_users']

ds['day'] = ds['day'].astype(str) + ':00:00:00'

fig = px.bar(

    ds, 

    x='day', 

    y="number_of_users", 

    orientation='v',

    title='Number of unique users per day', 

    width=800, 

    height=800

)

fig.show()
ds = df['day'].value_counts().reset_index()

ds.columns = ['day', 'count']

ds = ds.sort_values('count')

ds['day'] = ds['day'].astype(str) + ':00:00:00'

fig = px.bar(

    ds, 

    x='count', 

    y="day", 

    orientation='h',

    title='Tweets distribution over days present in dataset', 

    width=800, 

    height=800

)

fig.show()
df['hour'] = df['date'].dt.hour

ds = df['hour'].value_counts().reset_index()

ds.columns = ['hour', 'count']

ds['hour'] = 'Hour ' + ds['hour'].astype(str)

fig = px.bar(

    ds, 

    x="hour", 

    y="count", 

    orientation='v', 

    title='Tweets distribution over hours', 

    width=800

)

fig.show()
def split_hashtags(x): 

    return str(x).replace('[', '').replace(']', '').split(',')



tweets_df = df.copy()

tweets_df['hashtag'] = tweets_df['hashtags'].apply(lambda row : split_hashtags(row))

tweets_df = tweets_df.explode('hashtag')

tweets_df['hashtag'] = tweets_df['hashtag'].astype(str).str.lower().str.replace("'", '').str.replace(" ", '')

tweets_df.loc[tweets_df['hashtag']=='', 'hashtag'] = 'NO HASHTAG'

tweets_df
ds = tweets_df['hashtag'].value_counts().reset_index()

ds.columns = ['hashtag', 'count']

ds = ds.sort_values(['count'])

fig = px.bar(

    ds.tail(20), 

    x="count", 

    y='hashtag', 

    orientation='h', 

    title='Top 20 hashtags', 

    width=800, 

    height=700

)

fig.show()
df['tweet_length'] = df['text'].str.len()
fig = px.histogram(

    df, 

    x="tweet_length", 

    nbins=80, 

    title='Tweet length distribution', 

    width=800,

    height=700

)

fig.show()
ds = df[df['tweets_count']>=10]

ds = ds.groupby(['user_name', 'tweets_count'])['tweet_length'].mean().reset_index()

ds.columns = ['user_name', 'tweets_count', 'mean_length']

ds = ds.sort_values(['mean_length'])

fig = px.bar(

    ds.tail(40), 

    x="mean_length", 

    y="user_name", 

    color='tweets_count',

    orientation='h', 

    title='Top 40 users with the longest average length of tweet (at least 10 tweets)', 

    width=800, 

    height=800

)

fig.show()
ds = df[df['tweets_count']>=10]

ds = ds.groupby('user_name')['tweet_length'].mean().reset_index()

ds.columns = ['user_name', 'mean_length']

ds = ds.sort_values(['mean_length'])

fig = px.bar(

    ds.head(40), 

    x="mean_length", 

    y="user_name", 

    orientation='h', 

    title='Top 40 users with the shortest average length of tweet (at least 10 tweets)', 

    width=800, 

    height=800

)

fig.show()
def build_wordcloud(df, title):

    wordcloud = WordCloud(

        background_color='gray', 

        stopwords=set(STOPWORDS), 

        max_words=50, 

        max_font_size=40, 

        random_state=666

    ).generate(str(df))



    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')

    fig.suptitle(title, fontsize=16)

    fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
build_wordcloud(df['text'], 'Prevalent words in tweets for all dataset')
test_df = df[df['user_name']=='GlobalPandemic.NET']

build_wordcloud(test_df['text'], 'Prevalent words in tweets for GlobalPandemic.NET')
test_df = df[df['user_name']=='covidnews.ch']

build_wordcloud(test_df['text'], 'Prevalent words in tweets for covidnews.ch')
test_df = df[df['user_name']=='Open Letters']

build_wordcloud(test_df['text'], 'Prevalent words in tweets for Open Letters')
test_df = df[df['user_name']=='Hindustan Times']

build_wordcloud(test_df['text'], 'Prevalent words in tweets for Hindustan Times')
test_df = df[df['user_name']=='Blood Donors India']

build_wordcloud(test_df['text'], 'Prevalent words in tweets for Blood Donors India')
vec = TfidfVectorizer(stop_words="english")

vec.fit(df['text'].values)

features = vec.transform(df['text'].values)
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(features)
res = kmeans.predict(features)

df['Cluster'] = res

df
df[df['Cluster'] == 0].head(20)['text'].tolist()
df[df['Cluster'] == 1].head(20)['text'].tolist()
df['location'] = df['user_location'].str.split(',', expand=True)[1].str.lstrip().str.rstrip()

res = df.groupby(['day', 'location'])['text'].count().reset_index()
country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

res['alpha3'] = res['location']

res = res.replace({"alpha3": country_dict})



country_list = ['England', 'United States', 'United Kingdom', 'London', 'UK']



res = res[

    (res['alpha3'] == 'USA') | 

    (res['location'].isin(country_list)) | 

    (res['location'] != res['alpha3'])

]



gbr = ['England', 'UK', 'London', 'United Kingdom']

us = ['United States', 'NY', 'CA', 'GA']



res = res[res['location'].notnull()]

res.loc[res['location'].isin(gbr), 'alpha3'] = 'GBR'

res.loc[res['location'].isin(us), 'alpha3'] = 'USA'

res.loc[res['alpha3'] == 'USA', 'location'] = 'USA'

res.loc[res['alpha3'] == 'GBR', 'location'] = 'United Kingdom'

res = res.groupby(['day', 'location', 'alpha3'])['text'].sum().reset_index()

res
fig = px.choropleth(

    res, 

    locations="alpha3",

    hover_name='location',

    color="text",

    animation_frame='day',

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Tweets from different countries for every day',

    width=800, 

    height=600

)

fig.show()