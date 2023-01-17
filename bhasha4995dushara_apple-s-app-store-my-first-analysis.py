import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#import os
#print(os.listdir("../input"))
app_desc = pd.read_csv('../input/appleStore_description.csv')
app_desc.head()
mobile_app = pd.read_csv(f'../input/AppleStore.csv')
mobile_app.head()
app_desc.isnull().sum()
mobile_app.isnull().sum()
mobile_app = mobile_app.iloc[:,1:]
mobile_app.head()
mobile_app.info()
app_desc.info()
mobile_app.currency.value_counts()
for i in mobile_app.prime_genre.unique():
    print(i)
# For Games 
games = mobile_app[mobile_app['prime_genre'] == "Games"]
games.head()
mobile_app.user_rating.unique()
categories = mobile_app.prime_genre.value_counts()
plt.figure(figsize=(14,7))
sns.barplot(x=categories[:10].index,y=categories[:10].values)
plt.ylabel('App')
plt.xlabel('Categories')
plt.title('Top 10 App Categories',color = 'red',fontsize=15)
category = list(mobile_app.prime_genre.unique())
user_rating = []
for x in category:
    user_rating.append(mobile_app[mobile_app.prime_genre == x].user_rating.mean())

# DataFrame for category and user rating
rate = pd.DataFrame({'category': category,'user_rating':user_rating})
# set order basis on user rating
new_index = (rate['user_rating'].sort_values(ascending=False)).index.values
# for Valid indices
sorted_df_rating = rate.reindex(new_index)
#sorted_df_rating



plt.figure(figsize=(15,5))
sns.barplot(x=sorted_df_rating['category'], y=sorted_df_rating['user_rating'])
plt.xticks(rotation= 90)
plt.xlabel('Category')
plt.ylabel('Average User Rating')
plt.title('Categories and Average User Ratings')
user_rating_count = []
for i in category:
    user_rating_count.append(mobile_app[mobile_app.prime_genre == i].rating_count_tot.mean())
rating_count = pd.DataFrame({'category':category,'user_rating_count':user_rating_count})
new_index = rating_count['user_rating_count'].sort_values(ascending=False).index.values
count = rating_count.reindex(new_index)


plt.figure(figsize=(15,10))
sns.barplot(x=count['category'], y=count['user_rating_count'])
plt.xticks(rotation= 90)
plt.xlabel('Category')
plt.ylabel('Average User Rating Count')
plt.title('Categories and Average User Rating Counts')

    
# For Application
app = mobile_app[mobile_app['prime_genre'] != 'Games']
app.head()
user_rating = mobile_app.loc[:,["track_name","prime_genre","user_rating","rating_count_tot","price"]]
user_rating = user_rating.sort_values(by=["user_rating","rating_count_tot"],ascending=False)
user_rating.head()
plt.figure(figsize=(10,5))
sns.countplot(y = user_rating['prime_genre'])
user_rating_current_version = mobile_app.loc[:,["track_name","prime_genre","user_rating_ver","rating_count_ver","price"]]
user_rating_current_version = user_rating_current_version.sort_values(by = ["user_rating_ver","rating_count_ver"],ascending=False)
user_rating_current_version.head()
plt.figure(figsize=(50,20))
sns.countplot(user_rating_current_version['prime_genre'])
#free_game = mobile_app.loc[:,['track_name','currency','price','prime_genre','ver','user_rating']]
free_games = mobile_app[(mobile_app['prime_genre'] == 'Games') & (mobile_app['price'] == 0)]
free_games.head()
sns.countplot(user_rating['price']==0)
#paid_game = mobile_app.loc[:,['track_name','currency','price','prime_genre','ver','user_rating']]
paid_games = mobile_app[(mobile_app['prime_genre'] == 'Games') & (mobile_app['price'] != 0)]
paid_games.head(10)
sns.countplot(user_rating['price']!=0)
free_app = mobile_app[(mobile_app['prime_genre'] != 'Games') & (mobile_app['price'] == 0)]
free_app.head()
sns.countplot((user_rating['price']==0) & (user_rating['prime_genre'] != 'Games'))
paid_app = mobile_app[(mobile_app['prime_genre'] != 'Games') & (mobile_app['price'] != 0)]
paid_app.head()
sns.countplot((user_rating['price']!=0) & (user_rating['prime_genre'] != 'Games'))
n_user_rating = mobile_app['user_rating']
plt.figure(figsize=(15,5))
sns.countplot(n_user_rating)
plt.title("User Rating",fontsize=15)
free_games_user_rating = free_games[free_games['user_rating'] > 3.0]
free_games_user_rating.head()
plt.figure(figsize=(15,5))
sns.countplot(x='user_rating',data=free_games_user_rating)
plt.title("User rating is Greater than 3",fontsize=15)
paid_games_user_rating = paid_games[paid_games['user_rating'] > 3.0]
paid_games_user_rating.head()

plt.figure(figsize=(15,5))
sns.countplot(x = 'user_rating',data = paid_games_user_rating)
plt.title("User Rating is Greater than 3",fontsize=15)
free_app_user_rating = free_app[free_app['user_rating'] > 3.0]
free_app_user_rating.head()

plt.figure(figsize=(15,5))
sns.countplot(x = 'user_rating',data = free_app_user_rating)
plt.title("User Rating is Greater than 3",fontsize=15)
paid_app_user_rating = paid_app[paid_app['user_rating'] > 3.0]
paid_app_user_rating.head()

plt.figure(figsize=(15,5))
sns.countplot(x = 'user_rating',data = paid_app_user_rating)
plt.title("User Rating is Greater than 3",fontsize=15)
#free Games
#free_games["rating_count_tot"].mean()
free_games = free_games[free_games["rating_count_ver"]>free_games["rating_count_ver"].mean()]
plt.figure(figsize=(10,4))
sns.countplot(free_games["user_rating_ver"])
#paid 
paid_games = paid_games[paid_games["rating_count_ver"]>paid_games["rating_count_ver"].mean()]
plt.figure(figsize=(10,4))
sns.countplot(paid_games["user_rating_ver"])
