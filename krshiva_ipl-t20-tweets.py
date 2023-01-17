# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visulization
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join('/kaggle/input/ipl2020-tweets', 'IPL2020_Tweets.csv'))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read tweets  
iplt20_tweet = pd.read_csv('/kaggle/input/ipl2020-tweets/IPL2020_Tweets.csv')
iplt20_tweet.head()
iplt20_tweet.info()
# day of tweet  
iplt20_tweet.date.unique()
iplt20_tweet['date'] = pd.to_datetime(iplt20_tweet['date'])
iplt20_tweet['time'] = pd.to_datetime(iplt20_tweet['date']).dt.time
iplt20_tweet['time'].head()
#Tweet count by location 
iplt20_tweet.groupby(['user_location']).count()
# null values 
iplt20_tweet.isnull().sum()
#tweets Statistics 
iplt20_tweet.describe()
# retweet_count 
len(iplt20_tweet.is_retweet)
# correlation among tweet features
corr = iplt20_tweet.corr()
plt.figure(figsize=(10, 6))

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot= True)
most_followed_users = iplt20_tweet.drop_duplicates('user_name', keep='last')
most_followed_users_top_10 = most_followed_users.sort_values(by='user_followers').tail(10)
x = most_followed_users_top_10['user_name']
y = most_followed_users_top_10['user_followers']
plt.xlabel('Username')
plt.ylabel('Followers')
plt.title('Most followed user for Retweets')
#plt.xticks(range(10), x, rotation=60)
h=plt.bar(range(10), y, label='Most followed user for retweets')
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in h]

plt.xticks(xticks_pos, x,  ha='right', rotation=45)
plt.figure(figsize=(15, 12))
plt.show()
from fastai.text import *

 # specify path
path = Path('/kaggle/input/ipl2020-tweets', 'IPL2020_Tweets.csv')

