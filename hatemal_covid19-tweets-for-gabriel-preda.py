# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import lib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#read csv file
df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
df.head(3)
#check how amny rows and columns
df.shape
#check data types
df.dtypes
#chanage data column from object to datetime
df.date = pd.to_datetime(df['date'])
#create new columns ( year-month-day-hour)
df['year'] = df['date']
df['month'] = df['date']
df['day'] = df['date']
df['hour'] = df['date']
#check again columns and data type
df.dtypes
#identify my own table
df= df[['user_name','date','user_followers','user_location','user_verified','year','month','day','hour','source','hashtags','is_retweet','text']]
df.head(1)
# fill all new columns
df.date= df.date.dt.date
df.year = df.year.dt.year
df.month = df.month.dt.month
df.day = df.day.dt.day
df.hour = df.hour.dt.hour
#review result
df.head(1)
#confirm if there any none values in database
df.isna().count()
#analysis top 10 user who have followers
df_top_followers= df[['user_name','user_followers']].sort_values('user_followers',ascending=False)
#drop_duplicates users
df_top_followers = df_top_followers.drop_duplicates('user_name').head(10)
df_top_followers
#Top 10 user followers
df_top_followers.plot(kind='barh',x='user_name', y='user_followers',title='Top Users Followers')
plt.show()
#Top 10 user followers
sns.barplot(data=df_top_followers, x='user_followers', y='user_name',palette="Blues_d",capsize=1000).set_title('Top Users Followers')
plt.show()
df.user_location.value_counts().head(10).plot(kind='bar',x='user_location',y='Number of users', color='r',title='Top 10 Locations')
plt.show()
df.source.value_counts().head(10).plot(kind='barh', x='source' ,y='Number of users',title='Top 10 Source')
plt.show()
df.user_verified.value_counts().head(10).plot(kind='bar', x='user_verified' ,y='Number of users',title='Top User Verified')
plt.show()
df.date.value_counts().head(10).plot(kind='barh', x='date' ,y='Number of users', color='y',title='Top 10 Date')
plt.show()
df.month.value_counts().head(10).plot(kind='bar', x='month' ,y='Number of users',title='Top 10 month')
plt.show()
df.day.value_counts().head(10).plot(kind='barh', x='day' ,y='Number of users',title='Top 10 day')
plt.show()
df.hour.value_counts().head(10).plot(kind='bar', x='hour' ,y='Number of users',color='m',title='Top 10 hour')
plt.show()
df.hashtags.value_counts().head(10).plot(kind='barh', x='hashtags' ,y='Number of users',title='Top 10 Hashtags')
plt.show()
df.text.value_counts().head(5).plot(kind='barh', x='text' ,y='Number of users',title='Top 5 Sentence',color='g')
plt.show()
#Finally this my first work in Kaggel

