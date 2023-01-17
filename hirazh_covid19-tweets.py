
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline
tweets = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
tweets
tweets.describe()
tweets.info()
# get the location data
locations = tweets[['user_location', 'date']]
locations = locations[locations['user_location'].notnull()]
locations

top_locations = locations.groupby('user_location').user_location.count().to_frame('counts').reset_index()
top_locations = top_locations.sort_values(by='counts', ascending=False)
top_locations = top_locations[0:20]
#countries = top_locations['user_location'].to_list()

fig, ax = plt.subplots(figsize=(12,5))
ax = sns.barplot(data=top_locations,x=top_locations['user_location'], y=top_locations['counts'])
ax.set_xticklabels(top_locations['user_location'], rotation=90)

plt.title("Top locations", fontsize=18)
plt.xlabel("User Locations", fontsize=16)
plt.ylabel("Number of tweets", fontsize=16)