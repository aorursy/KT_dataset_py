import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
data.head()
data.tail()
data.columns   
data.shape
data.info()
data.describe()
data["user_verified"].unique()
is_verify = len(data[data["user_verified"]==True])
is_not_verify = len(data[data["user_verified"]==False])
print("Number of User Verified", is_verify)
print("Number of User Not Verified", is_not_verify)
print('Total_percentage_of_verified_user = ',(is_verify/data.shape[0])*100)

sns.countplot(data.user_verified)
plt.show()

def Top10(data):
    user_cols = ['user_name', 'user_location', 'source']
    for col in user_cols:
        data[col].value_counts().head(10).plot(kind = 'bar', figsize = (10,5))
        print(data[col].value_counts().head(10))
        plt.show()
        
Top10(data)
data['hashtags'] = data['hashtags'].fillna('[]')
data['hashtags_count'] = data['hashtags'].apply(lambda x: len(x.split(',')))
data.loc[data['hashtags'] == '[]', 'hashtags_count'] = 0
data.head(10)
data['hashtags_count'].describe()
ds = data.groupby('hashtags_count')['user_name'].count().reset_index()
ds.columns = ['hashtags_count', 'count']
ds = ds.sort_values(['count'])
ds['hashtags_count'] = ds['hashtags_count'].astype(str) + ' tags'
fig = sns.barplot(data=ds, x="count", y="hashtags_count")
fi

# ds = data[data['tweets_count']>10]
ds = data.groupby(['user_name'])['hashtags_count'].mean().reset_index()
ds.columns = ['user', 'mean_count']
ds = ds.sort_values(['mean_count'])
fig = sns.barplot(data=ds.tail(20), x="mean_count", y="user")
# fig.figure
data['date'] = pd.to_datetime(data['date']) 
data = data.sort_values(['date'])
data['day'] = data['date'].astype(str).str.split(' ', expand=True)[0]
data['time'] = data['date'].astype(str).str.split(' ', expand=True)[1]
data.head()
data['date'].head()
data['new_date'] = pd.to_datetime(data['date'])
data['new_date'].head()
data['year'] = data['new_date'].dt.year
data['month'] = data['new_date'].dt.month
data['day'] = data['new_date'].dt.day
data['dayofweek'] = data['new_date'].dt.dayofweek
data['hour'] = data['new_date'].dt.hour
data['minute'] = data['new_date'].dt.minute
data['dayofyear'] = data['new_date'].dt.dayofyear
data['date_only'] = data['new_date'].dt.date
data.groupby(['year', 'month'])['text'].count().plot(kind = 'bar', figsize = (15,5))
plt.show()
data.groupby( ['month', 'day'])['text'].count().plot(kind = 'bar', figsize = (15,5))
plt.show()
sns.heatmap(data.drop('is_retweet',axis=1).corr())
plt.title("Correlation in data")
plt.show()