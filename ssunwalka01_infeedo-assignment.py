# IMPORTING THE LIBRARIES



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# importing libraries

import seaborn as sns # for data visualisations

import matplotlib.pyplot as plt # for plotting graphs

%matplotlib inline

import warnings # to ignore warnings

warnings.filterwarnings("ignore")

from sklearn import metrics

import plotly.express as px

import json



from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob

from wordcloud import WordCloud,STOPWORDS

stopwords = set(STOPWORDS)





# Run this (by clicking run or pressing Shift+Enter)
!pip3 install geotext

!pip3 install country_converter
from geotext import GeoText

import country_converter as coco
# reading the data

df = pd.read_csv("../input/assignment/covid19_tweets.csv")
# printing the top 5 rows of df

df.head()
print("Columns in the dataset are : \n", df.columns)
print("Data types of Columns in the dataset are : \n", df.dtypes)
print("Shape of the Dataset : \n", df.shape)
print("Percentage wise :\n", df["user_verified"].value_counts(normalize=True)*100)   # percentage wise

df["user_verified"].value_counts(normalize=True).plot.bar(title = "Number of Verified Users")
print("Percentage wise :\n", df["is_retweet"].value_counts(normalize=True)*100)   # percentage wise

df["is_retweet"].value_counts(normalize=True).plot.bar(title = "Number of Retweets")
df['user_friends'].unique()
df['user_favourites'].unique()
corrmat = df.corr()

plt.subplots(figsize=(10, 10))

plt.title('Correlation Matrix')

sns.heatmap(corrmat, vmax=.9, annot = True ,square=True)
# checking missing data percentage in data

def misssing_values(df):

  total = df.isnull().sum().sort_values(ascending = False)

  percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

  missing_Data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



  return missing_Data
data_missing = misssing_values(df)

print("Missing values in Data\n\n\n", data_missing)
def unique_values(data_frame):

    unique_dataframe = pd.DataFrame()

    unique_dataframe['Columns/Features'] = data_frame.columns

    uniques = []

    for col in data_frame.columns:

        u = data_frame[col].nunique()

        uniques.append(u)

    unique_dataframe['No of Unique Values'] = uniques

    

    f, ax = plt.subplots(1,1, figsize=(10,5))#plt.figure(figsize=(10, 5))



    sns.barplot(x=unique_dataframe['Columns/Features'], y=unique_dataframe['No of Unique Values'], alpha=0.8)

    plt.title('Bar plot for no of unique values in each column')

    plt.ylabel('no of Unique values', fontsize=12)

    plt.xlabel('Columns/Features', fontsize=12)

    plt.xticks(rotation=90)

    plt.show()

    

    return unique_dataframe
# calling the function to plot and print

unique_df = unique_values(df)

print(unique_df)
covid_location_count=pd.DataFrame(df['user_location'].value_counts())

covid_location_count.head(10)
covid_location_count.reset_index(inplace=True) # reset index

covid_location_count.rename(columns={'index':'Location','user_location':'count'},inplace=True)

covid_location_count.sort_values(by='count',inplace=True,ascending=False)

Count_graph=px.bar(x='count',y='Location',data_frame=covid_location_count[:15],color='Location')

Count_graph.show()
covid_location_count.head()
Location_country=covid_location_count.copy() # saving a copy in case we make chages to this dataframe

Location_country['Location']=Location_country['Location'].apply(lambda x:x.replace(',',' '))

Location_country['Location']=Location_country['Location'].apply(lambda x:(GeoText(x).country_mentions))

Location_country.head()
Location_country.drop(Location_country[Location_country['Location']=='[]'].index,inplace=True)

Location_country['Location']=Location_country['Location'].apply(lambda x:(x.keys()))

Location_country['Location']=Location_country['Location'].apply(lambda x:list(x))

Location_country.drop(Location_country.index[Location_country.Location.map(len)==0],inplace=True)

Location_country['Location']=Location_country['Location'].apply(lambda x:str(x[0]))

agg_func={'count':'sum'}

Location_country=Location_country.groupby(['Location']).aggregate(agg_func)

Location_country.sort_values(by=['count'],ascending=False,inplace=True)

Location_country.reset_index(inplace=True)

Location_country.head(10)
Count_graph=px.bar(x='Location',y='count',data_frame=Location_country[:20],color='Location')

Count_graph.show()
cc = coco.CountryConverter()

Location_country['Location']=Location_country['Location'].apply(lambda x:cc.convert(names=x,to='ISO3'))
Count_graph_re=px.bar(x='Location',y='count',data_frame=Location_country[:20],color='Location')

Count_graph_re.show()
india_states = json.load(open("../input/country-state-geo-location/countries.geo.json", "r"))

fig = px.choropleth(

    Location_country,

    locations="Location",

    geojson=india_states,

    color="count",

    #hover_name="State or union territory",

    hover_data=["count"],

    title="Number of Tweets from each Country",

)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
df['date'] = pd.to_datetime(df['date']) 

df = df.sort_values(['date'])

df['day'] = df['date'].astype(str).str.split(' ', expand=True)[0]

df['time'] = df['date'].astype(str).str.split(' ', expand=True)[1]

df.head()

# splitting into day & time
df['hashtags'] = df['hashtags'].fillna('[]')

df['hashtags_count'] = df['hashtags'].apply(lambda x: len(x.split(',')))

df.loc[df['hashtags'] == '[]', 'hashtags_count'] = 0

df.head(10)
dss = df['user_name'].value_counts().reset_index()

dss.columns = ['user_name', 'tweets_count']

dss = dss.sort_values(['tweets_count'])

fig = px.bar(

    dss.tail(40), 

    x="tweets_count", 

    y="user_name", 

    orientation='h', 

    title='Top 40 users on # of tweets', 

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

    title='User Creation pattern on Twitter every year', 

    width=800, 

    height=600

)

fig.show()
ds = df['user_location'].value_counts().reset_index()

ds.columns = ['user_location', 'count']

ds = ds[ds['user_location']!='NA']

ds = ds.sort_values(['count'])

fig = px.bar(

    ds.tail(40), 

    x="count", 

    y="user_location", 

    orientation='h', title='Top 40 Users locations by no. of tweets', 

    width=800, 

    height=800

)

fig.show()
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
# no of unique users in one day

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
#Removing Stop Words

df['text'] = df['text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

df['text'].head() 
df['sentiment'] = ' '

df['polarity'] = None

for i,tweets in enumerate(df.text) :

    blob = TextBlob(tweets)

    df['polarity'][i] = blob.sentiment.polarity

    if blob.sentiment.polarity > 0 :

        df['sentiment'][i] = 'positive'

    elif blob.sentiment.polarity < 0 :

        df['sentiment'][i] = 'negative'

    else :

        df['sentiment'][i] = 'neutral'

df.head()
print(df.sentiment.value_counts())

sns.countplot(x='sentiment', data = df);
plt.figure(figsize=(10,6))

sns.distplot(df['polarity'], bins=30)

plt.title('Sentiment Distribution',size = 15)

plt.xlabel('Polarity',size = 15)

plt.ylabel('Frequency',size = 15)

plt.show();
count = pd.DataFrame(df.groupby('sentiment').sum())

count.head()
pla = df['source'][df['user_location'] == 'India'].value_counts().sort_values(ascending=False)

explode = (0, 0.1, 0, 0,0.01) 

plt.figure(figsize=(10,10))

pla[0:5].plot(kind = 'pie', title = '10 Most Tweet Sources used in India', autopct='%1.1f%%',shadow=True,explode = explode)



plt.show()