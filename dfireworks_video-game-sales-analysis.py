# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sns.set(style="whitegrid")

sns.set_palette("husl")
# Let's start from importing

video_game = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv",index_col="Rank")
# We should explore what we have in our data.

"""

We have some info about data's column names:

-Rank - Ranking of overall sales

-Name - The games name

-Platform - Platform of the games release (i.e. PC,PS4, etc.)

-Year - Year of the game's release

-Genre - Genre of the game

-Publisher - Publisher of the game

-NA_Sales - Sales in North America (in millions)

-EU_Sales - Sales in Europe (in millions)

-JP_Sales - Sales in Japan (in millions)

-Other_Sales - Sales in the rest of the world (in millions)

-Global_Sales - Total worldwide sales.

"""
video_game.info()
# Dropping nan values

video_game.dropna(how="any",inplace = True)

video_game.info()
video_game.shape 
# I want to see the dataset's column names

video_game.columns
video_game.head(10)
video_game.tail(10)
ZeroZeroOne=video_game[video_game["Global_Sales"]==0.01]

counted_zerozeroone = ZeroZeroOne["Global_Sales"].count()

print("Yes, there are {} games that have 0.01 million sales in Global_Sales column.".format(counted_zerozeroone))
platforms=video_game["Platform"]

platforms=platforms.drop_duplicates()

platforms=platforms.reset_index()

platforms=platforms.drop(columns="Rank")

counted_platforms=platforms["Platform"].count()

print("There are {} gaming platforms in this video game data".format(counted_platforms))
japan_sales= video_game["JP_Sales"]

sum_of_japan_sales=japan_sales.sum()

print(sum_of_japan_sales)

print("Sum of sales made in Japan is ${:.2f}(million)".format(sum_of_japan_sales))
jp_sales=video_game.sort_values("JP_Sales",ascending=False)

jp_sales= jp_sales.reset_index()
jp_sales = jp_sales[~(jp_sales["JP_Sales"] <= 0)]  
jp_latest = jp_sales[jp_sales["Year"]>=2015]

# Sorted by years from the latest to oldest(2015)

jp_latest = jp_latest.sort_values("Year",ascending=False)

jp_latest = jp_latest.reset_index()
ax=sns.countplot(y="Platform",data=jp_latest)

ax.set(xlabel="Count",ylabel="Platforms",title="Latest games in Japan by Platform")
ax=sns.barplot(x="JP_Sales",y="Platform",data=jp_latest,ci=None)

ax.set(xlabel="Video Game Sales in Japan(million $)",ylabel="Platforms",title="Top Selling Platforms from last 5 years")
jp_plats=jp_sales.groupby("Platform")["JP_Sales"].sum()

jp_plats_sorted=jp_plats.sort_values(ascending=False)
jp_plats_sorted.head(3)
jp_heads=jp_plats_sorted.head(5)

jp_heads=jp_heads.reset_index()
ax=sns.barplot(y="Platform",x="JP_Sales",data=jp_heads)

ax.set(xlabel="Video Game Sales in Japan(million $)",ylabel="Platforms",title="Best Selling Gaming Platforms in Japan")
eu_sales=video_game.sort_values("EU_Sales",ascending=False)

eu_sales=eu_sales.reset_index()

eu_sales = eu_sales[~(eu_sales["EU_Sales"] <= 0)] 
# I want to select last 5 year's games

eu_latest = eu_sales[eu_sales["Year"]>=2015]

# Sorted by years from the latest to oldest

eu_latest = eu_latest.sort_values("Year",ascending=False)

eu_latest = eu_latest.reset_index() 
ax=sns.countplot(y="Platform",data=eu_latest)

ax.set(xlabel="Count",ylabel="Platforms",title="Latest games in Europe by Platform")
ax=sns.barplot(x="EU_Sales",y="Platform",data=eu_latest,ci=None)

ax.set(xlabel="Video Game Sales in Europe(million $)",ylabel="Platforms",title="Top Selling Platforms from last 5 years")
eu_plats=eu_sales.groupby("Platform")["EU_Sales"].sum()

eu_plats_sorted=eu_plats.sort_values(ascending=False)
eu_plats_sorted.head(3)

eu_heads=eu_plats_sorted.head(5)

eu_heads=eu_heads.reset_index()
ax=sns.barplot(y="Platform",x="EU_Sales",data=eu_heads)

ax.set(xlabel="Video Game Sales in Europe (million $)",ylabel="Platforms",title="Best Selling Gaming Platforms in Europe")
na_sales=video_game.sort_values("NA_Sales",ascending=False)

na_sales=na_sales.reset_index()

na_sales = na_sales[~(na_sales["NA_Sales"] <= 0)] 
# I want to select last 5 year's games

na_latest = na_sales[na_sales["Year"]>=2015]

# Sorted by years from the latest to oldest

na_latest = na_latest.sort_values("Year",ascending=False)

na_latest = na_latest.reset_index() 
ax=sns.countplot(y="Platform",data=na_latest)

ax.set(xlabel="Count",ylabel="Platforms",title="Latest games in North America by Platform")
ax=sns.barplot(x="NA_Sales",y="Platform",data=na_latest,ci=None)

ax.set(xlabel="Video Game Sales in North America (million $)",ylabel="Platforms",title="Top Selling Platforms from last 5 years")
na_plats=na_sales.groupby("Platform")["NA_Sales"].sum()

na_plats_sorted=na_plats.sort_values(ascending=False)
na_plats_sorted.head(3)

na_heads=na_plats_sorted.head(5)

na_heads=na_heads.reset_index()
ax=sns.barplot(y="Platform",x="NA_Sales",data=na_heads)

ax.set(xlabel="Video Game Sales in North America (million $)",ylabel="Platforms",title="Best Selling Gaming Platforms in Europe")
video_game.describe()
global_sales=video_game["Global_Sales"]

sum_of_global_sales = global_sales.sum()

print(sum_of_global_sales)

print("Sum of sales made in the entire world is ${:.2f}(millions)".format(sum_of_global_sales))
dark_souls=video_game[video_game["Name"].isin(["Dark Souls","Dark Souls II","Dark Souls III","Bloodborne"])]

dark_souls=dark_souls.reset_index()

dark_souls=dark_souls.sort_values(["Name","Rank"],ascending=True)
print(dark_souls[["Rank","Name","Platform"]])
ax=sns.scatterplot(x="Rank",y="Platform",hue="Name",data=dark_souls)

ax.set(xlabel="Rankings",ylabel="Platforms",title="Souls Games' rankings by Platform")
gta=video_game[video_game["Name"]=="Grand Theft Auto V"]

gta=gta[gta["Platform"]=="PS3"]

gtaps3=gta["Global_Sales"]
gtaps3=float(gtaps3)

more_than_gta=video_game[video_game["Global_Sales"]>gtaps3]

print(more_than_gta[["Name","Publisher"]])
new=video_game[video_game["Year"]==2020]

print(new[["Name","Publisher","Global_Sales"]])