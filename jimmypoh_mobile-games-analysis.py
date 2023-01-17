# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



games = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")
games.describe()
games.info()
games.head()
# I will drop some features that are not useful for my analysis

games.drop(["URL","ID","Icon URL","Subtitle","Age Rating","Languages","Size"], axis=1, inplace=True)
games.isnull().sum()

# there are some missing values
# I will replace "Average User Rating" missing values with its minimum value

games["Average User Rating"] = games["Average User Rating"].fillna(games["Average User Rating"].min())
# same goes with "User Rating Count" 

games["User Rating Count"] = games["User Rating Count"].fillna(games["User Rating Count"].min())
games[games["Price"].isnull()]

# most of the games with missing price do have in-app purchases.

# I will replace "Price" missing valus with 0 

# game companies will likely to charge in-app purchases while distributing the games for free in order to attract more players

# at the same time, game companies will earn through in-app purchases
games["Price"] = games["Price"].fillna(0)
# for "In-app Purchases" missing values, I will replace them with 0 if the game companies charge any price for the game

# if the game is free-to-play, I will replace missing values with minimum in-app-purchases.



for i in games["Price"]:

    if i > 0:

        games["In-app Purchases"].fillna(0, inplace=True)

    else:

        games["In-app Purchases"].fillna(0.99, inplace=True)
games.isnull().sum()

# there is no missing value
# I will change "Original Release Date" and "Current Version Release Date" into datetime formats

games["Original Release Date"] = pd.to_datetime(games["Original Release Date"])

games["Current Version Release Date"] = pd.to_datetime(games["Current Version Release Date"])
# I will create a new column "Days of Release" to show how many days those games have been released starting from their original release date up to latest version release date

games["Days of Release"] = games["Current Version Release Date"] - games["Original Release Date"]
games.head()
############################## Exploratory Data Analysis #################################



df_ten_developer = pd.DataFrame(games["Developer"].value_counts().sort_values(ascending=False).reset_index())

df_ten_developer.columns = ["Developer","Counts"]

df_ten_developer = df_ten_developer.head(10)



plt.figure(figsize=(16,8))

sns.barplot(x="Developer", y="Counts", data=df_ten_developer)

plt.xticks(rotation=80)

plt.show()

# these are the top 10 developers that have created the most mobile games
df_rating_count = games.groupby("Developer").agg({"User Rating Count":"mean"}).sort_values("User Rating Count", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="Developer", y="User Rating Count", data=df_rating_count.head(10))

plt.xticks(rotation=80)

plt.show()

# it seems that "Supercell" developer has the highest average of user rating count
df_genres = games.groupby(["Developer","Genres"]).agg({"User Rating Count":"mean"}).sort_values("User Rating Count", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="Developer", y="User Rating Count",hue="Genres", data=df_genres.head(10))

plt.xticks(rotation=80)

plt.show()

# it seems that "Supercell" company is quite popular for its action, strategy and entertainment games
df_rating = games.groupby(["Developer","Genres"]).agg({"Average User Rating":"mean"}).sort_values("Average User Rating", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="Developer", y="Average User Rating",hue="Genres", data=df_rating.head(10))

plt.xticks(rotation=80)

plt.show()

# it seems few game genres of "Dustin Allen" has received high "Average User Rating"
df_release = games.groupby("Developer").agg({"Days of Release":"max"}).sort_values("Days of Release", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="Developer", y="Days of Release", data=df_release.head(10))

plt.xticks(rotation=80)

plt.show()

# "Spiffcode, Inc" has produced the game with longest lifespan by far
games[games["Days of Release"] == games["Days of Release"].max()]

# it seems that this particular game is quite famous among players.
df_genres_days = games.groupby("Genres").agg({"Days of Release":"max"}).sort_values("Days of Release", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="Genres", y="Days of Release", data=df_genres_days.head(10))

plt.xticks(rotation=80)

plt.show()

# it seems that "Strategy" game has the longest lifespan

# perhaps nowadays players prefer strategy games 
# Text Mining on "Description"

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



stop_words = stopwords.words("english")

lem = WordNetLemmatizer()
def cleaning_text(i):

    i = re.sub("[^A-Za-z]+"," ",i).lower()

    i = re.sub("[0-9]+"," ", i)

    lem_words = []

    for x in i.split(" "):

        x = lem.lemmatize(x)

        lem_words.append(x)

    words = []

    for z in lem_words:

        if z not in stop_words:

            words.append(z)

    w = []

    for j in words:

        if len(j) > 3:

            w.append(j)

    return(" ".join(w))
wordnet = games["Description"].apply(cleaning_text)

wordnet = " ".join(wordnet)
from wordcloud import WordCloud



wordcloud_games = WordCloud(background_color="black",height=1400, width=1800).generate(wordnet)

plt.figure(figsize=(16,8))

plt.imshow(wordcloud_games)

# "player battle" has caught my attention as it has coincides with "strategy" game as being the most popular game genre

# perhaps players prefer those strategy games that allow them to test their skills against other players.
# As a conclusion, I believe nowadays players prefer to play strategy games which allow them to fight against other players.

# This is quite make sense as MOBA(Multiplayer Online Battle Arena) such as Dota is the top game genre right now.