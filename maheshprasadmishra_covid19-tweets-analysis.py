import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
!pip install geotext
c_df = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
c_df.head(10)
c_df.shape
c_df.info()
c_df.describe()
100*c_df.isnull().sum()/len(c_df)
## Dropping the NA values, as imputing values would not be right for this dataset.
c_df.dropna(inplace = True)
## Na Check
100*c_df.isnull().sum()/len(c_df)
## Top 10 user_location or places of tweets.

tweet_location = c_df['user_location'].value_counts()
tweet_loc = tweet_location[:10]
tweet_loc
tw_loc = pd.DataFrame(tweet_loc)
tw_loc = tw_loc.reset_index()
tw_loc.columns = ['Location', 'counts']
tw_loc
plt.figure(figsize = (15,10))
sns.barplot(x = tw_loc.Location,y = tw_loc.counts, data = tw_loc)
tweet_location
Loc_count=pd.DataFrame(c_df['user_location'].value_counts())
Loc_count.reset_index(inplace = True)
Loc_count.rename(columns={'index':'Location','user_location':'count'},inplace=True)
Loc_count.sort_values(by='count',inplace=True,ascending=False)
Loc_count
from geotext import GeoText
location=Loc_count.loc[2]['Location']
print(GeoText(location).countries)
#!pip install geotext
loc_country=Loc_count.copy()
loc_country['Location']=loc_country['Location'].apply(lambda x:x.replace(',',' '))
loc_country['Location']=loc_country['Location'].apply(lambda x:(GeoText(x).country_mentions))
loc_country
loc_country.drop(loc_country[loc_country['Location']=='[]'].index,inplace=True)
loc_country['Location']=loc_country['Location'].apply(lambda x:(x.keys()))
loc_country['Location']=loc_country['Location'].apply(lambda x:list(x))
loc_country
loc_country.drop(loc_country.index[loc_country.Location.map(len)==0],inplace=True)
loc_country
loc_country['Location']=loc_country['Location'].apply(lambda x:str(x[0]))
loc_country
agg_func={'count':'sum'}
loc_country=loc_country.groupby(['Location']).aggregate(agg_func)
loc_country
loc_country.sort_values(by=['count'],ascending=False,inplace=True)
loc_country.reset_index(inplace=True)
loc_country
loc_country20 = loc_country.head(20)
loc_country20
plt.figure(figsize = (15,10))
sns.barplot(x = 'Location', y = 'count' ,data = loc_country20)
plt.title('Top 20 countries with highest count of tweets', fontweight = 'bold')
## Data Analysis of the Date Column

date_df = c_df.copy()
date_df['date'].value_counts()
date_df['Dates'] = pd.to_datetime(date_df['date']).dt.date
date_df['Time'] = pd.to_datetime(date_df['date']).dt.time
date_count = date_df['Dates'].value_counts()
date_count = date_df['Dates'].value_counts().reset_index()
date_count = pd.DataFrame(date_count)
date_count.columns = ['Date', 'counts']

date_count
plt.figure(figsize = (20,10))
sns.lineplot(x =date_count.Date, y = date_count.counts, color = 'red', marker = 'o')
plt.title('Number of Tweet per day')
c_df.head()
ht_count = c_df['hashtags'].value_counts()
ht_count
def htg(x):
    x = str(x)
    x = x.replace('[', '')
    x = x.replace(']', '')
    x = x.split(',')
    return x

## The hashtags are not clear and also not unique words, so the code below transformed it into unique words.

ht_count = c_df['hashtags'].value_counts().reset_index()
ht_count = pd.DataFrame(ht_count)
ht_count.columns = ['Hashtags', 'counts']
ht_count['Hashtags'] = ht_count['Hashtags'].apply(lambda x:htg(x))
ht_count = ht_count.explode('Hashtags')
ht_count['Hashtags'] = ht_count['Hashtags'].str.lower()
ht_count['Hashtags'] = ht_count['Hashtags'].str.replace(" ","")
ht_count['Hashtags'] = ht_count['Hashtags'].str.replace("'","")
ht_count10 = ht_count['Hashtags'].value_counts().reset_index()
ht_count10 = ht_count10[0:10]
ht_count10.rename(columns = {'index':'HashTag', 'Hashtags':'Count'}, inplace = True)
ht_count10
plt.figure(figsize = (20,10))
sns.barplot(x = ht_count10.HashTag, y = ht_count10.Count, data = ht_count10)
plt.title('Top 10 Hashtags on Twitter', fontweight = 'bold', fontsize='15')
c_df.head()
c_df['source'].value_counts()
t_source = c_df['source'].value_counts().reset_index()
t_source = pd.DataFrame(t_source)
t_source.columns = ['Source of Tweet', 'counts']
t_source = t_source[t_source['counts']>=300]
t_source
plt.figure(figsize = (30,15))
sns.barplot(x = 'Source of Tweet', y = 'counts', data = t_source)
plt.title('Sources of Tweets with count more than 300', fontweight = 'bold')
rts = c_df['is_retweet'].value_counts().reset_index()

rts = pd.DataFrame(rts)
rts.columns = ['RT: True or False', 'counts']
rts
##finding out the date of the first tweet in dataset
date_col = date_df['Dates']
date_col.reset_index()
date_col = pd.DataFrame(date_col)
date_col.sort_values(by='Dates',inplace=True)
date_col['Dates'].iloc[0:1]

bef_date = date_df[date_df['user_created'] < '2020-07-24']
aft_date = date_df[date_df['user_created'] >= '2020-07-24']
len1 = bef_date.shape[0]
len2 = aft_date.shape[0]
bef_perc = len1/(len1+len2)
aft_perc = len2/(len1+len2)
data = [['Before 1st Tweet',bef_perc],['After 1st Tweet',aft_perc]]
acc_creation = pd.DataFrame(data, columns = ['Category', 'Percent'])
acc_creation

c_df.head()
username = c_df['user_name'].value_counts().reset_index()
username = pd.DataFrame(username)
username.columns = ['Twitter UserName', 'Number of Tweets']
username = username.head(15)
username
plt.figure(figsize = (30,10))
sns.barplot(x = 'Twitter UserName', y = 'Number of Tweets', data = username)
c_df['user_verified'].value_counts()
plt.figure(figsize = (10,10))
sns.countplot('user_verified', data = c_df)
new = c_df[['user_name', 'user_followers']].copy()
new = new.drop_duplicates(subset = ['user_name'])

new = new.sort_values(by = ['user_followers'], ascending = False)
new15 = new.head(15)
new15
plt.figure(figsize = (30,15))
sns.barplot(x = 'user_name', y = 'user_followers', data = new15)
plt.xlabel('User Name', fontweight = 'bold')
plt.ylabel('No of Follower', fontweight = 'bold')
plt.title('Top 15 Twitter usernames by follower count')
year_df = c_df.copy()
year_df['Year'] = pd.to_datetime(year_df['user_created']).dt.year
year_df['Year'].value_counts()
plt.figure(figsize = (20,10))
sns.countplot(year_df['Year'], data = year_df)
hour1 = c_df.copy()
hour1['hour_of_day'] = pd.to_datetime(hour1['date']).dt.hour
hour1.head()
hour2 = hour1['hour_of_day'].value_counts().reset_index()
hour2 = pd.DataFrame(hour2)
hour2.columns = ['Hour of Day', 'count of tweets per hour']
hour2 = hour2.sort_values(by = ['count of tweets per hour'], ascending = False)
hour2

plt.figure(figsize = (15,8))
sns.barplot(x = 'Hour of Day',y = 'count of tweets per hour', data = hour2)
plt.title('Tweet count by Hour of the Day', fontweight = 'bold')
from wordcloud import WordCloud, STOPWORDS
def word_cloud_func(c_df):
    word_cloud = WordCloud(
    background_color = 'black',
    stopwords = set(STOPWORDS),
    max_words = 50,
    max_font_size = 40,
    
    ).generate(str(c_df))
    fig = plt.figure(
    figsize = (20, 20),
    facecolor = 'k',
    edgecolor = 'k')
    plt.axis('off')
    
    fig.subplots_adjust(top=2.3)
    plt.imshow(word_cloud)
    plt.show()
word_cloud_func(c_df['text'])
temp_df = c_df[c_df['user_name']=='covidnews.ch']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='GlobalPandemic.NET']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='Blood Donors India']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='Hindustan Times']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='IANS Tweets']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='Shashi Tharoor']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='NDTV']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='World Health Organization (WHO)']
word_cloud_func(temp_df['text'])
temp_df = c_df[c_df['user_name']=='OTV']
word_cloud_func(temp_df['text'])
word_cloud_func(c_df['user_description'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
kc_df = c_df.copy()
vec = TfidfVectorizer(stop_words="english")
vec.fit(kc_df['text'].values)
features = vec.transform(kc_df['text'].values)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features)
temp = kmeans.predict(features)
kc_df['Cluster'] = temp
kc_df
len(kc_df[kc_df['Cluster'] == 0])
len(kc_df[kc_df['Cluster'] == 1])
word_cloud_func(kc_df[kc_df['Cluster'] == 0]['text'])
word_cloud_func(kc_df[kc_df['Cluster'] == 1]['text'])
kmeans1 = KMeans(n_clusters=4, random_state=0)
kmeans1.fit(features)
temp1 = kmeans1.predict(features)
kc_df['Cluster4'] = temp1
kc_df
word_cloud_func(kc_df[kc_df['Cluster4'] == 0]['text'])
word_cloud_func(kc_df[kc_df['Cluster4'] == 1]['text'])
word_cloud_func(kc_df[kc_df['Cluster4'] == 2]['text'])
word_cloud_func(kc_df[kc_df['Cluster4'] == 3]['text'])




