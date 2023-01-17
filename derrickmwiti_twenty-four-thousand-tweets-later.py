# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/tweets.xlsx',sheet='tweets')
df.head()
df.info()
pd.isnull(df).any()
df.describe()
df[df['retweets']==79537]
plt.figure(figsize=(12,8))

sns.countplot(data=df,y='username')
toptweeps= df.groupby('username')[['tweet ']].count()

toptweeps.sort_values('tweet ',ascending=False)[:10]
topretweets= df.groupby('username')[['retweets']].sum()

topretweets.sort_values('retweets',ascending=False)[:10]
corpus = ' '.join(df['tweet '])

corpus = corpus.replace('.', '. ')

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
mest = df[df['username']=='MESTAfrica']

corpu = ' '.join(df['tweet '])

corpu = corpu.replace('.', '. ')

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpu)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
tony = df[df['username']=='TonyElumeluFDN']

corp = ' '.join(df['tweet '])

corp = corp.replace('.', '. ')

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corp)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
mest.describe()
mest[mest['retweets']==2157]
df2 = df

df2['date'] = df2['created_at'].map(lambda x: x.split(' ')[0])

df2['time'] = df2['created_at'].map(lambda x: x.split(' ')[-1])

del df2['created_at']

df2.head()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df2= df[['tweet_id','date','time','tweet ','retweets','username']]

df2.head()
df2['month'] = df2['date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(df2['month'].value_counts()).reset_index()

month_df.columns = ['month', 'tweets']
plt.figure(figsize=(12,6))

plt.title("All Tweets Per Month")

sns.barplot(x='month', y='tweets', data=month_df, order=month_order)
def getday(x):

    year, month, day = (int(i) for i in x.split('-'))    

    answer = datetime.date(year, month, day).weekday()

    return day_order[answer]

df['day'] = df['date'].apply(getday)

day_df = pd.DataFrame(df['day'].value_counts()).reset_index()

day_df.columns = ['day', 'tweets']

plt.figure(figsize=(12,6))

plt.title("All Tweets Per Day")

sns.barplot(x='day', y='tweets', data=day_df, order=day_order)
mesting = df2[df2['username']=='MESTAfrica']

month_mest = pd.DataFrame(mesting['month'].value_counts()).reset_index()

month_mest.columns = ['month', 'tweets']



plt.figure(figsize=(12,6))

plt.title("MESTAfrica  Tweets Per Month")

sns.barplot(x='month', y='tweets', data=month_mest, order=month_order)

month_mest.head()
month_mest['tweets'].sum()
def getday(x):

    year, month, day = (int(i) for i in x.split('-'))    

    answer = datetime.date(year, month, day).weekday()

    return day_order[answer]

df['day'] = mesting['date'].apply(getday)

day_df = pd.DataFrame(df['day'].value_counts()).reset_index()

day_df.columns = ['day', 'tweets']

plt.figure(figsize=(12,6))

plt.title("MESTAfrica Tweets Per Day")

sns.barplot(x='day', y='tweets', data=day_df, order=day_order)
retweets_df = mesting.groupby('month')['retweets'].sum().reset_index()

plt.figure(figsize=(12,6))

plt.title("MEST Retweets per month")

sns.barplot(x="month",y="retweets",data=retweets_df,order=month_order)

retweets_gen = df2.groupby('month')['retweets'].sum().reset_index()

plt.figure(figsize=(12,6))

plt.title("All Retweets per month")

sns.barplot(x="month",y="retweets",data=retweets_gen,order=month_order)
def getday(x):

    year, month, day = (int(i) for i in x.split('-'))    

    answer = datetime.date(year, month, day).weekday()

    return day_order[answer]

df['day'] = df['date'].apply(getday)

day_df = df.groupby('day')['retweets'].sum().reset_index()



plt.figure(figsize=(12,6))

plt.title("All Reweets Per Day")

sns.barplot(x='day', y='retweets', data=day_df, order=day_order)
mest_days = df[df['username']=='MESTAfrica']
df_mest = df.groupby('day')['retweets'].sum().reset_index()



plt.figure(figsize=(12,6))

plt.title("MESTAfrica Reweets Per Day")

sns.barplot(x='day', y='retweets', data=day_df, order=day_order)
tweets_retweets = df.groupby('username')['retweets'].sum().reset_index()

plt.figure(figsize=(12,6))

plt.title('Retweets per organization')

sns.barplot(x='retweets',y='username',data=tweets_retweets)
by_retweets = df.groupby('username')['retweets'].sum().reset_index()

by_retweets.head()
by_tweets = df.groupby('username')['tweet '].count().reset_index()

by_tweets.head()
merged_df = pd.merge(by_retweets,by_tweets,how='left',left_on='username',right_on='username')

merged_df.head()

sns.jointplot(data=merged_df,x='tweet ',y='retweets',color='g')

plt.title('Relationship between no of tweets and no of retweets')