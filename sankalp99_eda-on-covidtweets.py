# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')

df.head()
df.shape
df.isnull().sum()
df.dropna(inplace=True)

df.head()
df.shape
df.isnull().sum()
#No of unique tweet locations in the dataset.

len(df['user_location'].value_counts())
tweetRegions = df['user_location'].value_counts().reset_index()

tweetRegions.sort_values(by='user_location',ascending=False,inplace=True)

tweetRegions.rename(columns = {'index':'Location', 'user_location':'Count'}, inplace = True)
#Top 10 Locations with hightest no of tweets

tweetRegions = tweetRegions[:10]

tweetRegions
fig,ax = plt.subplots(1,1, figsize=(18,8))

sns.set(style="whitegrid")

sns.barplot(x=tweetRegions.Location,y=tweetRegions.Count,palette='Blues_r')

plt.title("Tweets vs Location",fontsize=15)

plt.xticks(fontsize=12)
dates = df.copy()

dates['date'] = pd.to_datetime(dates['date'])

dates['day'] = dates['date'].dt.date

day_count = dates['day'].value_counts().reset_index()

day_count.rename(columns = {'index':'Date', 'day':'Count'}, inplace = True)

day_count.sort_values(by='Date',ascending=True, inplace=True)

day_count
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.set(style="whitegrid")

sns.lineplot(x=day_count.Date,y=day_count.Count,color='orange',marker='o')

plt.title("No. of Tweets By Date",fontsize=15)

plt.xticks(rotation=45, fontsize=12)
day_names={

    0:'Monday',

    1:'Tueday',

    2:'Wednesday',

    3:'Thursday',

    4:'Friday',

    5:'Saturday',

    6:'Sunday'

}

days = df.copy()

days['date'] = pd.to_datetime(days['date'])

days['day'] = days['date'].dt.dayofweek

day_weeks = days['day'].value_counts().reset_index()

day_weeks.rename(columns = {'index':'Day', 'day':'Count'}, inplace = True)

day_weeks.sort_values(by='Day',inplace=True)

day_weeks['Day'] = day_weeks['Day'].apply(lambda d:day_names[d])

day_weeks['Percent'] = (day_weeks['Count']/sum(day_weeks['Count'])*100)

day_weeks
fig,ax = plt.subplots(1,1, figsize=(10,10))

sns.set(style="whitegrid")

sns.barplot(x=day_weeks.Day,y=day_weeks.Percent,palette='nipy_spectral',alpha=0.8)

plt.title("Tweets Vs Day of the Week",fontsize=15)

plt.xlabel('Day of the Week')

plt.ylabel('% of tweets')

plt.xticks(fontsize=12)
#Top 10 HASHTAGS



def makeList(x):

    x = str(x)

    x = x.replace('[', '')

    x = x.replace(']', '')

    x = x.split(',')

    return x



df_tags = df.copy()

df_tags['hashes'] = df_tags['hashtags'].apply(lambda x:makeList(x))

df_tags = df_tags.explode('hashes')

df_tags['hashes'] = df_tags['hashes'].str.lower()

df_tags['hashes'] = df_tags['hashes'].str.replace(" ","")

df_tags['hashes'] = df_tags['hashes'].str.replace("'","")

tags = df_tags['hashes'].value_counts().reset_index()

tags = tags[0:10]

tags.rename(columns = {'index':'HashTag', 'hashes':'Count'}, inplace = True)

tags
fig,ax = plt.subplots(1,1, figsize=(10,10))

sns.set(style="whitegrid")

sns.barplot(x=tags.HashTag,y=tags.Count,palette='OrRd_r')

plt.title("Top 10 Hashtags with frequency",fontsize=15)

plt.xticks(rotation=45,fontsize=12)
#Sources with atleast 100 tweets



sources = df['source'].value_counts().reset_index()

sources.rename(columns = {'index':'Source', 'source':'Count'}, inplace = True)

sources = sources[sources['Count']>=100]

sources
fig,ax = plt.subplots(1,1, figsize=(30,10))

sns.set(style="whitegrid")

sns.barplot(x=sources.Source,y=sources.Count,palette='Greens_r')

plt.title("Tweets vs Source",fontsize=15)

plt.xticks(rotation=45,fontsize=12)
retweet = df['is_retweet'].value_counts().reset_index()

retweet.rename(columns={'index':'Retweeted','is_retweet':'count'},inplace=True)

retweet
#Date of the 1st tweet in dataset

user = df.copy()

user['user_created'] = pd.to_datetime(user['user_created'])

user['date'] = pd.to_datetime(user['date'])

user.sort_values(by='date',inplace=True)

user['date'].iloc[0:1]
before = user[user['user_created'] < '2020-07-24']

after = user[user['user_created'] >= '2020-07-24']

len1 = before.shape[0]

len2 = after.shape[0]

bef_perc = len1/(len1+len2)

aft_perc = len2/(len1+len2)

data = [['Before COVID-19',bef_perc],['After COVID-19',aft_perc]]

accounts = pd.DataFrame(data, columns = ['Tag', 'Percent'])

accounts
fig,ax = plt.subplots(1,1, figsize=(5,10))

sns.set(style="whitegrid")

sns.barplot(x=accounts.Tag,y=accounts.Percent,palette='prism',alpha=0.7)

plt.title("Tweets vs Source",fontsize=15)

plt.xticks(fontsize=12)