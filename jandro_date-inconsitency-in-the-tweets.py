# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
!pip install git+https://github.com/abenassi/Google-Search-API
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('classic')
%matplotlib inline
import seaborn as sns
from google import google

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw = pd.read_csv('../input/Donald-Tweets!.csv')
raw.describe()
pattern = re.compile("^\\\".*@realDonaldTrump.*\"$")

retweets_index = raw['Tweet_Text'].apply(lambda d: 'RT @' in d)
citations_index = raw['Tweet_Text'].apply(lambda d: None != re.match(pattern, d))
only_tweets_index = ~retweets_index & ~ citations_index
print(f"all : {raw.size} \njust tweets : {raw[only_tweets_index].size} \nretweets : {raw[retweets_index].size} \ncites : {raw[citations_index].size}")
# Just for show purposes
raw[retweets_index].head(2)
raw[citations_index].head(2)
columns = [c for c in raw.columns[-2:]]
columns.append('Tweet_Id')
raw[columns].describe()
for c in columns:
    print(f'{c} has {len(raw[c].unique())} different values')
# I shall removed these columns, and format the rest of the columns consequently
raw = raw.drop(raw.columns[-2:],axis="columns")
raw = raw.drop('Tweet_Id',axis="columns")

raw['Date'] = raw['Date'].apply(lambda d:pd.to_datetime('-'.join(reversed(d.split('-')))))
raw.loc[:,'week_day'] = raw['Date'].apply(lambda d:d.weekday)
def get_activity_df(original):
    activity = original.copy()
    activity = activity['Date'].value_counts()
    activity = pd.DataFrame({
        'date':activity.index,
        'count':activity
    })
    activity.loc[:, 'day_of_week'] = \
        activity.iloc[:]['date'].apply(lambda d: pd.to_datetime(d).weekday())
    return activity
weekdays = ['Monday', 'Tusday', 'Wendsday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.set()
fig=plt.figure(figsize=(20,5))
gs=gridspec.GridSpec(1,2) # 2 rows, 3 columns

ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[0,1]) # First row, second column

sb1 = sns.boxplot(x="day_of_week", y="count", data=get_activity_df(raw), ax=ax1)
sb2 = sns.boxplot(x="day_of_week", y="count", data=get_activity_df(raw[only_tweets_index]), ax=ax2)

ax1.set_title('With retweets and citations')
ax1.set_xticklabels(weekdays)
ax2.set_title('Without retweets and citations')
_ = ax2.set_xticklabels(weekdays)
sb3 = sns.relplot(x="date", y="count", data=get_activity_df(raw), kind='line', aspect=4)
concatenated = pd.concat([get_activity_df(raw).assign(dataset='set1'), get_activity_df(raw[only_tweets_index]).assign(dataset='set2')])

sb3 = sns.relplot(x="date", y="count", data=concatenated, kind='line', aspect=4, hue='dataset')
activity = get_activity_df(raw)
activity[activity['count']>40]
urls = [
    "https://twitter.com/realDonaldTrump/status/788932604460347392",
    "https://twitter.com/realDonaldTrump/status/788930678255517696",
    "https://twitter.com/realDonaldTrump/status/788922462733869056"
]
pd.set_option('display.max_colwidth', -1)
raw.loc[raw['Tweet_Url'].isin(urls), ['Date','Tweet_Text','Tweet_Url']]
pattern = re.compile('^[a-z,A-Z][a-z,A-Z][a-z,A-Z] [0-9][0-9], [0-9][0-9][0-9][0-9]')
date = lambda entry: pd.to_datetime(entry[:12])

done = 0
dates = []
for entry in raw.loc[only_tweets_index & (raw['Date'] == '20-10-16'),:].itertuples(index=True, name='Pandas'):
    url = entry.Tweet_Url
    result = google.search(url)
    if result and len(result)>=1:
        text = result[0].description[:]
        match = re.match(pattern, text)
        if match is not None:
            dates.append(dict(url=url, tweet=entry.Tweet_Text, date=date(text)))
    done += 1