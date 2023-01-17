import numpy as np #linear algebra
import pandas as pd #data processing

import seaborn as sns #visualization
import matplotlib
import matplotlib.pyplot as plt #visualization
%matplotlib inline
plt.style.use('bmh')

from datetime import datetime
users = pd.read_csv('../input/users.csv')

users.head()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(users[np.isfinite(users.statuses_count)].statuses_count, ax=ax[0])
sns.distplot(users[np.isfinite(users.followers_count)].followers_count, ax=ax[1])
plt.show()
form = '%a %b %d %H:%M:%S %z %Y'
users = users.assign(date = users.created_at.map(
    lambda x: datetime.strptime(str(x), form).date() if x is not np.nan else None))
users = users.set_index(pd.DatetimeIndex(users.date))
monthseries = users.groupby(by=[users.index.month]).count()
YearSeries = users.groupby([users.index.year]).count()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.barplot(monthseries.index, monthseries.id, ax=ax[0])
sns.barplot(YearSeries.index, YearSeries.id, ax=ax[1])
plt.show()
TimeSeries = users.groupby([users.index.year, users.index.month]).count()
plt.figure(figsize=(12,6))
TimeSeries.id.plot()
plt.xticks(rotation=45)
plt.ylabel('Number of New Users')
plt.xlabel('Year, Month')
plt.show()
tweets = pd.read_csv('../input/tweets.csv')
form = '%Y-%m-%d %H:%M:%S'
tweets = tweets.assign(date = tweets.created_str.map(
    lambda x: datetime.strptime(str(x), form).date() if x is not np.nan else None))
tweets = tweets.set_index(pd.DatetimeIndex(tweets.date))
timeseries = tweets.groupby([tweets.index.year, tweets.index.month]).count()
plt.figure(figsize=(12,6))
timeseries.user_id.plot()
plt.xticks(rotation=45)
plt.ylabel('Number of New Tweets')
plt.xlabel('Year, Month')
plt.show()
tags = tweets.text.copy()

# This code extracts where the retweet is from, as it follows a "RT @XXXXX:" format
retweets = tags.str.extract('(@.*:)', expand=True)

# Gets rid of website links
tags = tags.replace('https.*$','',regex=True)
# Gets rid of twitter handles
tags = tags.replace('@.*:','',regex=True)
# Gets rid of RT
tags = tags.replace('RT|amp|co','',regex=True)
from wordcloud import WordCloud, STOPWORDS

text = ' '.join([str(x) for x in tags.values])

wc = WordCloud(stopwords=STOPWORDS,background_color='white',max_words=200,scale=3).generate(text)
plt.figure(figsize=(15,15))
plt.axis('off')
plt.imshow(wc)
plt.show()
retweets = retweets.replace(':.*','',regex=True)
print(retweets[0].value_counts().describe())
user_retweeted = retweets[0].value_counts().index[~pd.isnull(retweets[0].value_counts().index)]
retweeted_user = ['@'+x for x in tweets.user_key.value_counts().index]

cascade = [x for x in retweeted_user if x in user_retweeted]
# easy method to remove the @ symbol again and make a clean user_key set
cascade = pd.Series(cascade).replace('@','',regex=True)
network = tweets.user_key.value_counts()
# wish I knew of this previously, kept trying to join two pd.Series which pandas isn't a fan of
network = network[network.index.intersection(cascade.values)]
network
print('Total number of retweets from users contained within the dataset: {}'.format(network.values.sum()))
print('Percentage of total dataset: {}'.format(network.values.sum()/len(tweets)))