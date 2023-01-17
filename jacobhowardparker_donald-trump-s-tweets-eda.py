import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly 

import plotly.express as px





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data=pd.read_csv('../input/trump-tweets/realdonaldtrump.csv')
data.shape
data.head(5)
print(data.mentions.mode())

print(data.hashtags.mode())

big_mention=pd.DataFrame([mentions for mentions in data.mentions if len(str(mentions))>20])

big_hashtag=pd.DataFrame([hashtag for hashtag in data.hashtags if len(str(hashtag))>20])

big_mention.head(), big_hashtag.head()

#looking at numerical values

data.describe()
#Looking at length of tweets

data['tweet_length'] = data['content'].apply(str)

data['tweet_length']=data['tweet_length'].apply(len)
data['tweet_length'].describe()
data.iloc[data.loc[data.tweet_length==data.tweet_length.max()].index[0], 2]
#shortest tweet 

print(data.iloc[data.loc[data.tweet_length==data.tweet_length.min()].index[0], 2])
print('Total favorites: ', data['favorites'].sum())
data['date']=data['date'].apply(pd.to_datetime)

fig=px.line(data, x='date', y='favorites', title='Favorites Time Series')

fig['data'][0]['line']['color']='blue'

fig.update_xaxes(rangeslider_visible=True)

fig.show()

print('Total retweets: ', data['retweets'].sum())
fig=px.line(data, x='date', y='retweets', title='Retweets Time Series')

fig['data'][0]['line']['color']='red'

fig.update_xaxes(rangeslider_visible=True)

fig.show()
#Most favorited tweet: 



print(data.iloc[data.loc[data.favorites==data.favorites.max()].index[0], 2])

#Most retweeted tweet:

print(data.iloc[data.loc[data.retweets==data.retweets.max()].index[0], 2])
#Slice the dataframe by year and store in dict

yr_data={}

for year in range(2009,2021):

    yr_data[year]=data[(data['date'] >= str(year)+'-01-01') & (data['date']<=str(year)+'-12-31')]

    yr_data[year]=yr_data[year].reset_index() #resets the index to start from 0 from each slice

    yr_data[year]=yr_data[year].drop('index', axis=1)

#create a new dataframe with key callouts for each year

years=list(range(2009,2021))

data_year=pd.DataFrame(data={'Year':years})

favorites_max, favorites_max_content, favorites_mean, retweets_max,retweets_max_content, retweets_mean=[],[],[],[],[],[]

favorites_max_content=[]

for year in years:

    favorites_max.append(yr_data[year].favorites.max())

    favorites_max_content.append(yr_data[year].iloc[yr_data[year].loc[yr_data[year].favorites==yr_data[year].favorites.max()].index[0],2])

    favorites_mean.append(int(yr_data[year].favorites.mean()))

    retweets_max.append(yr_data[year].retweets.max())

    retweets_max_content.append(yr_data[year].iloc[yr_data[year].loc[yr_data[year].retweets==yr_data[year].retweets.max()].index[0],2])

    retweets_mean.append(int(yr_data[year].retweets.mean()))

data_year['favorites_max']=favorites_max

data_year['favorites_max_content']=favorites_max_content

data_year['favorites_mean']=favorites_mean

data_year['retweets_max']=retweets_max

data_year['retweets_max_content']=retweets_max_content

data_year['retweets_mean']=retweets_mean

data_year
fig=px.scatter(data_year, x='Year', y='favorites_max', size='favorites_max', hover_name='Year', hover_data=['favorites_max_content'])

fig.add_scatter(x=data_year['Year'],y=data_year['favorites_mean'], mode='lines' , name='mean_favorites')

fig.update_layout(showlegend=True)

fig.show()
#fig=px.line(data_year, x='Year', y='retweets_max', hover_name='Year', hover_data=['retweets_max_content'])

fig=px.scatter(data_year, x='Year', y='retweets_max', size='retweets_max', hover_name='Year', hover_data=['retweets_max_content'])

fig.add_scatter(x=data_year['Year'],y=data_year['retweets_mean'], mode='lines' , name='mean_retweets')

fig.update_layout(showlegend=True)

fig.show()
#we create a function that will split our hashtag and mentions data and return a dict of individual hashtags and mentions with frequency

def freq(column):

    freq_dict={}

    for point in column: 

        sep_points=point.split(',')

        for point in sep_points:

            if point not in freq_dict.keys():

                freq_dict[point]=1

            else:

                freq_dict[point]+=1

    return freq_dict

        
#for entry in mentions column

#split into list

#for each entry in list, append to a master list/df

data['mentions']=data['mentions'].fillna('None')

data.mentions.apply(str)

mention_dict=freq(data.mentions)

mention_dict.pop('None')

    
len(mention_dict)
mentions=pd.Series(mention_dict, name='num_mentions')

mentions.index.name='User'

mentions.sort_values(ascending=False, inplace=True)



fig=px.bar(mentions, x=mentions[19::-1], y=mentions.index[19::-1], labels={'x':'num_mentions','y':''})

fig.show()
data['hashtags']=data['hashtags'].fillna('None')

hashtags_dict=freq(data.hashtags)

hashtags_dict.pop('None')
hashtags=pd.Series(hashtags_dict, name='num_uses')

hashtags.index.name='hashtag'

hashtags.sort_values(ascending=False, inplace=True)



fig=px.bar(hashtags, x=hashtags[19::-1], y=hashtags.index[19::-1], labels={'x':'num_uses','y':''})

fig.show()
import re

def clean(text):

    text=str(text).lower() #lowercase

    text = re.sub('(https?:\/\/)(www\.)?\S+', '', text) #removes links

    text=re.sub('(pic\.)\S+','',text) #removes links to twitter pics/gifs

    text=re.sub(r'\@(\s)?\S+','', text) #removes mentions

    text=re.sub(r'\#\S+','',text) #removes hashtags

    text=re.sub(r'[^\w\s]',' ',text)  #remove punctuation (adds a space)

    text=re.sub(r'\s+', ' ', text)   #removes doublespace

    return text
from wordcloud import WordCloud

data['clean_text']=data.content.apply(lambda x: clean(x))

text=" ".join(tweet for tweet in data['clean_text'])

wordcloud=WordCloud(max_font_size=2000, max_words=2000, background_color='white',random_state=42).generate(text)

plt.figure(figsize=(20,20))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.title('Most popular words')

plt.show()



data['len_clean_text']=data.clean_text.apply(len)

data['len_clean_text'].describe()
fig1=px.scatter(data[data['favorites']>200000],x='len_clean_text',y='favorites', title='Length vs Favorites')

fig1.show()

fig2=px.scatter(data[data['retweets']>75000],x='len_clean_text', y='retweets' , title='Length vs Retweets')

fig2.show()
#add day of week column

data['day_of_week']=data['date'].dt.day_name()

data[data['day_of_week']=='Monday'].shape[0]

tweet_days={}

for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:

    tweet_days[day]=data[data['day_of_week']==day].shape[0]

tweet_days=pd.Series(tweet_days, name='num_tweets')

tweet_days.index.name='day'

fig=px.bar(tweet_days, x=tweet_days.index, y=tweet_days, labels={'x':'Day','y':'Number of tweets'})

fig.show()
data['per_month']=data.date.dt.to_period('M')



def group_sum(date):

    return data[data['per_month']==date].shape[0]



monthly_tweets=pd.DataFrame(data['per_month'].unique(), columns=['date_month'])

monthly_tweets['num_tweets']=monthly_tweets['date_month'].apply(group_sum)

monthly_tweets
fig=px.bar(x=monthly_tweets['date_month'].apply(str), y=monthly_tweets['num_tweets'], hover_name=monthly_tweets['date_month'].apply(str), hover_data={'Daily':round(monthly_tweets['num_tweets']/monthly_tweets['date_month'].dt.days_in_month,1)}, title='Monthly tweets over time')



fig.show()
text_jan15=" ".join(tweet for tweet in data[data['per_month']=='2015-01'].clean_text)

wordcloud=WordCloud(max_font_size=2000, max_words=2000, background_color='white',random_state=42).generate(text_jan15)

plt.figure(figsize=(20,20))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.title('Most popular words')

plt.show()