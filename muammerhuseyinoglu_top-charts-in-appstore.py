import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/AppleStore.csv")
df.head()
df.info()
ratings = df.loc[:,["track_name","prime_genre","user_rating","rating_count_tot","price"]]
ratings = ratings.sort_values(by=["user_rating","rating_count_tot"],ascending=False)
ratings.head()
plt.figure(figsize=(10,5))
sns.countplot(y= ratings["prime_genre"])
sns.countplot(ratings["price"]==0)
plt.figure(figsize=(10,4))
sns.countplot(ratings["user_rating"])
ratings["rating_count_tot"].mean()
ratings = ratings[ratings["rating_count_tot"]>ratings["rating_count_tot"].mean()]
plt.figure(figsize=(10,4))
sns.countplot(ratings["user_rating"])
top_free_games = ratings[(ratings["prime_genre"]=="Games") & (ratings["price"]==0)]
top_free_games.head(10)
top_paid_games = ratings[(ratings["prime_genre"]=="Games") & (ratings["price"]!=0)]
top_paid_games.head(10)
top_free_apps = ratings[(ratings["prime_genre"]!="Games") & (ratings["price"]==0)]
top_free_apps.head(10)
top_paid_apps = ratings[(ratings["prime_genre"]!="Games") & (ratings["price"]!=0)]
top_paid_apps.head(10)
trend_ratings = df.loc[:,["track_name","prime_genre","user_rating_ver","rating_count_ver","price"]]
trend_ratings = trend_ratings[trend_ratings["rating_count_ver"]>trend_ratings["rating_count_ver"].mean()]
plt.figure(figsize=(10,4))
sns.countplot(trend_ratings["user_rating_ver"])
trend_ratings = trend_ratings.sort_values(by=["user_rating_ver","rating_count_ver"],ascending=False)
trend_ratings.head()
top_trending_free_games = trend_ratings[(trend_ratings["prime_genre"]=="Games") & (trend_ratings["price"]==0)]
top_trending_free_games.head(10)
top_trending_paid_games = trend_ratings[(trend_ratings["prime_genre"]=="Games") & (trend_ratings["price"]!=0)]
top_trending_paid_games.head(10)
top_trending_free_apps = trend_ratings[(trend_ratings["prime_genre"]!="Games") & (trend_ratings["price"]==0)]
top_trending_free_apps.head(10)
top_trending_paid_apps = trend_ratings[(trend_ratings["prime_genre"]!="Games") & (trend_ratings["price"]!=0)]
top_trending_paid_apps.head(10)