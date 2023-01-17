import numpy as np

import pandas as pd

import plotly.express as px

from iso3166 import countries

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")

df.head()
df['date'] = pd.to_datetime(df['date']) 

df = df.sort_values(['date'])

df['day'] = df['date'].astype(str).str.split(' ', expand=True)[0]
def split_hashtags(x): 

    return str(x).replace('[', '').replace(']', '').split(',')



df['hashtag'] = df['hashtags'].apply(lambda row : split_hashtags(row))

df = df.explode('hashtag')

df['hashtag'] = df['hashtag'].astype(str).str.lower().str.replace("'", '').str.replace(" ", '')

df = df[df['hashtag']!='nan']
hashtags = df.groupby(['day', 'hashtag'])['user_name'].count().reset_index()

hashtags.columns = ['day', 'hashtag', 'count']
def plot_hashtags_by_day(data, hashtag):

    data = data[data['hashtag']==hashtag]

    fig = px.line(

        data, 

        x='day', 

        y='count', 

        orientation='v', 

        title='Dynamic of hashtag "' + hashtag + '"' , 

        width=800

    )

    fig.show()
plot_hashtags_by_day(hashtags, 'corona')
plot_hashtags_by_day(hashtags, 'covid19')
plot_hashtags_by_day(hashtags, 'coronavirus')
plot_hashtags_by_day(hashtags, 'hydroxychloroquine')
plot_hashtags_by_day(hashtags, 'vaccine')
plot_hashtags_by_day(hashtags, 'pandemic')
hashtags_country = df.groupby(['day', 'hashtag', 'user_location'])['user_name'].count().reset_index()

hashtags_country.columns = ['day', 'hashtag', 'location', 'count']



hashtags_country['location'] = hashtags_country['location'].str.split(',', expand=True)[1].str.lstrip().str.rstrip()



country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

hashtags_country['alpha3'] = hashtags_country['location']

hashtags_country = hashtags_country.replace({"alpha3": country_dict})



country_list = ['England', 'United States', 'United Kingdom', 'London', 'UK']



hashtags_country = hashtags_country[

    (hashtags_country['alpha3'] == 'USA') | 

    (hashtags_country['location'].isin(country_list)) | 

    (hashtags_country['location'] != hashtags_country['alpha3'])

]



gbr = ['England', 'United Kingdom', 'London', 'UK']

us = ['United States', 'NY', 'CA', 'GA']



hashtags_country = hashtags_country[hashtags_country['location'].notnull()]

hashtags_country.loc[hashtags_country['location'].isin(gbr), 'alpha3'] = 'GBR'

hashtags_country.loc[hashtags_country['location'].isin(us), 'alpha3'] = 'USA'



hashtags_country.loc[hashtags_country['alpha3'] == 'USA', 'location'] = 'USA'

hashtags_country.loc[hashtags_country['alpha3'] == 'GBR', 'location'] = 'United Kingdom'

hashtags_country = hashtags_country.groupby(['day', 'hashtag', 'location', 'alpha3'])['count'].sum().reset_index()

hashtags_country
def plot_hashtag_map(data, hashtag):

    data = data[data['hashtag']==hashtag]

    fig = px.choropleth(

        data, 

        locations="alpha3",

        hover_name="hashtag",

        color="count",

        animation_frame="day",

        projection="natural earth",

        color_continuous_scale=px.colors.sequential.Plasma,

        title='Dynamic of hashtag "' + hashtag + '"' ,

        width=800, 

        height=600

    )

    fig.show()
plot_hashtag_map(hashtags_country, 'covid19')
plot_hashtag_map(hashtags_country, 'coronavirus')
plot_hashtag_map(hashtags_country, 'vaccine')
top20_hashtags_list = hashtags.groupby(['hashtag'])['count'].sum().reset_index().sort_values('count', ascending=False).head(20)['hashtag'].tolist()

hdf = hashtags[hashtags['hashtag'].isin(top20_hashtags_list)]

hdf
fig = px.bar(

    hdf, 

    x="count", 

    y="hashtag", 

    animation_frame="day", 

    orientation='h', 

    title='Dynamic of top 20 hashtags', 

    width=800, 

    height=700

)

fig.show()
df['country'] = df['user_location'].str.split(',', expand=True)[1].str.lstrip().str.rstrip()



country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

df['alpha3'] = df['country']

df = df.replace({"alpha3": country_dict})



country_list = ['England', 'United States', 'United Kingdom', 'London', 'UK']



df = df[

    (df['alpha3'] == 'USA') | 

    (df['country'].isin(country_list)) | 

    (df['country'] != df['alpha3'])

]



df = df[df['country'].notnull()]

df.loc[df['country'] == 'England', 'alpha3'] = 'GBR'

df.loc[df['country'] == 'United States', 'alpha3'] = 'USA'

df.loc[df['country'] == 'United Kingdom', 'alpha3'] = 'GBR'

df.loc[df['country'] == 'London', 'alpha3'] = 'GBR'

df.loc[df['country'] == 'UK', 'alpha3'] = 'GBR'

df.loc[df['country'] == 'NY', 'alpha3'] = 'USA'

df.loc[df['country'] == 'CA', 'alpha3'] = 'USA'

df.loc[df['country'] == 'GA', 'alpha3'] = 'USA'



df.loc[df['alpha3'] == 'USA', 'country'] = 'USA'

df.loc[df['alpha3'] == 'GBR', 'country'] = 'United Kingdom'

df
res = df.groupby(['country', 'day'])['text'].count().reset_index()

top5list = res.groupby(['country'])['text'].sum().reset_index().sort_values('text', ascending=False).head(5)['country'].tolist()
data = res[res['country'].isin(top5list)]

fig = px.line(

    data, 

    x="day", 

    y="text", 

    title='Dynamic of top 5 countries', 

    color='country'

)

fig.show()
res = df.groupby(['country', 'day'])['text'].count().reset_index()

top12list = res.groupby(['country'])['text'].sum().reset_index().sort_values('text', ascending=False).head(12)['country'].tolist()

data = res[res['country'].isin(top12list)]

fig = px.bar(

    data, 

    x="day", 

    y="text", 

    color='country', 

    title='Dynamic for top 12 countries'

)

fig.show()
udf = df.groupby(['day', 'country', 'alpha3', 'user_name'])['user_location'].count().reset_index().drop(['user_location'], axis=1)

udf = udf.groupby(['day', 'country', 'alpha3'])['user_name'].count().reset_index()

udf
fig = px.scatter_geo(

    udf, 

    locations="alpha3",  

    size="user_name", 

    animation_frame="day",

    projection="natural earth", 

    width=800, 

    height=600, 

    title='Dynamic of number of users'

)

fig.show()
df['tweet_len'] = df['text'].str.len()

data = df.groupby('day')['tweet_len'].mean().reset_index()
fig = px.line(

    data, 

    x="day", 

    y="tweet_len", 

    title='Average len of tweets'

)

fig.show()
df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")

ds = df.groupby('user_name')['user_location'].count().reset_index()

ds.columns = ['user_name', 'count']

ds = ds.sort_values(['count'])

most_active = ds['user_name'].tail(1).tolist()

df['date'] = pd.to_datetime(df['date']) 

df = df.sort_values(['date'])

df['day'] = df['date'].astype(str).str.split(' ', expand=True)[0]

ds = df[df['user_name'].isin(most_active)]

ds = ds.groupby(['user_name', 'day', 'user_followers'])['user_friends'].count().reset_index()[['user_name', 'day', 'user_followers']]

fig = px.line(

    ds, 

    x="day", 

    y="user_followers", 

    color='user_name', 

    title='Followers dynamic'

)

fig.show()
def split_hashtags(x): 

    return str(x).replace('[', '').replace(']', '').split(',')



df['hashtag'] = df['hashtags'].apply(lambda row : split_hashtags(row))

df = df.explode('hashtag')

df['hashtag'] = df['hashtag'].astype(str).str.lower().str.replace("'", '').str.replace(" ", '')

df = df[df['hashtag']!='nan']

trump = df[df['hashtag']=='trump']
ds = trump['user_name'].value_counts().reset_index()

ds.columns = ['user_name', 'tweets']

ds = ds.sort_values(['tweets'])

top5users = ds.tail(5)['user_name'].unique().tolist()

fig = px.bar(

    ds.tail(20), 

    x="tweets", 

    y="user_name", 

    orientation='h', 

    title='Top 20 users by total number of tweets with hashtag "trump"'

)

fig.show()
ds = trump.groupby(['day', 'user_name'])['hashtag'].count().reset_index()

ds.columns = ['day', 'user_name', 'tweets']

ds = ds[ds['user_name'].isin(top5users)]

fig = px.line(

    ds, 

    x="day", 

    y="tweets", 

    color='user_name', 

    title='Dynamic for top 5 users'

)

fig.show()
data = trump.groupby('day')['user_name'].count().reset_index()

data.columns = ['day', 'count']

fig = px.line(

    data, 

    x="day", 

    y="count", 

    title='Dynamic for "trump" hashtag'

)

fig.show()
trump_country = trump.groupby(['user_location'])['user_name'].count().reset_index()

trump_country.columns = ['location', 'count']



trump_country['location'] = trump_country['location'].str.split(',', expand=True)[1]

trump_country['location'] = trump_country['location'].str.lstrip()

trump_country['location'] = trump_country['location'].str.rstrip()



country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

trump_country['alpha3'] = trump_country['location']

trump_country = trump_country.replace({"alpha3": country_dict})



trump_country = trump_country[

    (trump_country['alpha3'] == 'USA') | 

    (trump_country['location'] == 'England') | 

    (trump_country['location'] == 'United States') | 

    (trump_country['location'] == 'United Kingdom') |  

    (trump_country['location'] == 'London') | 

    (trump_country['location'] == 'UK') | 

    (trump_country['location'] != trump_country['alpha3'])

]





trump_country = trump_country[trump_country['location'].notnull()]

trump_country.loc[trump_country['location'] == 'England', 'alpha3'] = 'GBR'

trump_country.loc[trump_country['location'] == 'United States', 'alpha3'] = 'USA'

trump_country.loc[trump_country['location'] == 'United Kingdom', 'alpha3'] = 'GBR'

trump_country.loc[trump_country['location'] == 'London', 'alpha3'] = 'GBR'

trump_country.loc[trump_country['location'] == 'UK', 'alpha3'] = 'GBR'

trump_country.loc[trump_country['location'] == 'NY', 'alpha3'] = 'USA'

trump_country.loc[trump_country['location'] == 'CA', 'alpha3'] = 'USA'

trump_country.loc[trump_country['location'] == 'GA', 'alpha3'] = 'USA'



trump_country.loc[trump_country['alpha3'] == 'USA', 'location'] = 'USA'

trump_country.loc[trump_country['alpha3'] == 'GBR', 'location'] = 'United Kingdom'

trump_country = trump_country.groupby(['location', 'alpha3'])['count'].sum().reset_index()
fig = px.bar(

    trump_country, 

    x="location", 

    y="count", 

    title='Countries that used hashtag "trump"',

    width=800

)

fig.show()
fig = px.scatter_geo(

    trump_country, 

    locations="alpha3", 

    size="count", 

    projection="natural earth", 

    width=800, 

    height=600, 

    title='Countries that used hashtag "trump"'

)

fig.show()
def build_wordcloud(df, title):

    wordcloud = WordCloud(

        background_color='black', 

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
build_wordcloud(trump['text'], 'Prevalent words in tweets with hashtag "trump"')