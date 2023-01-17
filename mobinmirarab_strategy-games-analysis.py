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
# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from wordcloud import WordCloud



# display settings

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)
# initializing dataframe

df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")
# dataframe shape

df.shape
# column info

list(df.columns)
# columns info

df.info()
# printing the first 5 rows

df.head()
# dropping the columns URL, ID, Icon URL

df.drop(columns=["URL","ID","Icon URL"], inplace=True)
# finding the no. of missing values in each column

df.isna().sum()
# taking an initial look at the values and scanning for junk values

df["Subtitle"].dropna().head()
# no. of rows in each rating bracket

df["Average User Rating"].value_counts()
# summary statistics

df["Average User Rating"].describe()
# plotting the boxplot to understand the outliers

sns.boxplot(df["Average User Rating"])

plt.show()
# taking an initial look at the values and scanning for junk values

df["User Rating Count"].dropna().sort_values().head()
# summary statistics

df["User Rating Count"].describe()
# boxplot and distribution plot

fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.distplot(df["User Rating Count"].dropna(), ax=ax[0])

sns.boxplot(df["User Rating Count"].dropna(), ax=ax[1])

plt.show()
# checking the different price brackets

df["Price"].value_counts().sort_index()
# summary statistics

df["Price"].describe()
# splitting the string into float values and storing them as a list

df["In-app Purchases"] = df["In-app Purchases"].dropna().map(lambda x: list(float(i) for i in x.split(", ")))
# splitting the string and storing the values as a list

df["Languages"] = df["Languages"].dropna().map(lambda x: x.split(", "))
# filling the missing value

df["Size"].fillna(method="ffill", inplace=True)
# converting size in bytes to mega-bytes

df["Size"] = df["Size"].map(lambda x: round(x/(1024 * 1024), 2))

df["Size"].head()
# splitting the string and storing the values as a list

df["Genres"] = df["Genres"].map(lambda x: x.split(", "))

df["Genres"].head()
# converting string to date

df["Original Release Date"] = df["Original Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))
# converting string to date

df["Current Version Release Date"] = df["Current Version Release Date"].map(lambda x: datetime.strptime(x, "%d/%m/%Y"))
# categorizing price

df["Price Range"] = df["Price"].dropna().map(lambda x: "Free" if x == 0.00 else("Low Price" if 0.99 <= x <= 4.99 else("Medium Price" if 5.99 <= x <= 19.99 else "High Price")))

df["Price Range"].value_counts()
df["Total In-app Purchases"] = df["In-app Purchases"].dropna().map(lambda x: sum(x))

df["Total In-app Purchases"].dropna().value_counts().head()
df["Game Genre"] = df[df["Primary Genre"] == "Games"]["Genres"].map(lambda x: x[1])

df["Game Genre"].head()
df["Release Year"] = df["Original Release Date"].map(lambda x: x.strftime("%Y"))

df["Release Month"] = df["Original Release Date"].map(lambda x: x.strftime("%m"))
df.info()
top_genres = list(df["Primary Genre"].value_counts().head(10).index)
df[df["Primary Genre"].isin(top_genres)]["Primary Genre"].value_counts().plot.bar(figsize=(8,5))

plt.title("Bar plot of primary genre wise apps")

plt.show()
def word_cloud(list_variable):

    fig, ax = plt.subplots(1,3, figsize=(15,4))

    for i, variable in enumerate(list_variable):

        corpus = df[variable].dropna()

        if variable not in ("Genres"):

            corpus = corpus.map(lambda x: x.replace(",", "").split(" "))

            corpus = corpus.map(lambda x: [word for word in x if len(word) > 3])

        corpus = ",".join(word for word_list in corpus for word in word_list)

        wordcloud = WordCloud(max_font_size=None, background_color="white", collocations=False, width=1500, height=1500).generate(corpus)

        ax[i].imshow(wordcloud)

        ax[i].set_title(variable)

        ax[i].axis("off")

    plt.show()



word_cloud(["Genres", "Subtitle", "Description"])
df[df["Primary Genre"].isin(top_genres)].groupby("Primary Genre")["Average User Rating"].agg("mean").sort_values().plot.bar(figsize=(8,6))

plt.title("Primary Genre wise Average User Rating")

plt.show()
ct_genre_agerating = pd.crosstab(df[df["Primary Genre"].isin(top_genres)]["Primary Genre"], df["Age Rating"], normalize=0)

ct_genre_agerating.plot.bar(stacked=True, figsize=(8,5))

plt.title("Primary Genre repartition by Age Rating")

plt.show()
df["Age Rating"].value_counts().plot.pie(autopct="%1.1f", explode=[0,0,0.1,0], figsize=(6,6))

plt.title("Age Rating wise app proportions")

plt.show()
plt.figure(figsize=(12,6))

sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y="Size")

plt.xticks(rotation=90)

plt.title("Primary Genre wise average size of apps")

plt.show()
df["Price Range"].dropna().value_counts().plot.pie(autopct="%1.1f", explode=[0,0.1,0,0], figsize=(6,6))

plt.title("Price Range wise proportion of apps")

plt.show()
ct_agerating_pricerange = pd.crosstab(df["Age Rating"], df["Price Range"], normalize=0)

ct_agerating_pricerange.plot.bar(stacked=True, figsize=(8,5))

plt.xticks(rotation=0)

plt.title("Age Rating repartioned by Price Range")

plt.show()
plt.figure(figsize=(8,5))

sns.barplot(data=df, x="Price Range", y="Average User Rating")

plt.title("Average user rating in each price range")

plt.show()
fig, ax = plt.subplots(2,2, figsize=(15,10))

sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0,0]).set_xticklabels(ax[0,0].get_xticklabels(), rotation=45)

sns.barplot(data=df, x="Age Rating", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0,1])

sns.barplot(data=df[df["Primary Genre"].isin(top_genres)], x="Primary Genre", y="Total In-app Purchases", ax=ax[1,0]).set_xticklabels(ax[1,0].get_xticklabels(), rotation=90)

sns.barplot(data=df, x="Age Rating", y="Total In-app Purchases", ax=ax[1,1])

ax[0,0].set_title("Average no. of in-app purchase in each genre")

ax[0,1].set_title("Average no. of in-app purchase in each age rating")

ax[1,0].set_title("Average value of in-app purchase in each genre")

ax[1,1].set_title("Average value of in-app purchase in each age rating")

plt.show()
# creating list of top game genres

top_game_genre = list(df["Game Genre"].value_counts().head(11).index)
fig, ax = plt.subplots(1,2, figsize=(15,5))

sns.countplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", ax=ax[0]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)

sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y="Average User Rating", ax=ax[1]).set_xticklabels(ax[1].get_xticklabels(), rotation=90)

ax[0].set_title("Count of games in each genre")

ax[1].set_title("Average rating of games in each genre")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(18,5))

sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y=df["In-app Purchases"].dropna().map(lambda x: len(x)), ax=ax[0]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)

sns.barplot(data=df[df["Game Genre"].isin(top_game_genre)], x="Game Genre", y="Total In-app Purchases", ax=ax[1]).set_xticklabels(ax[0].get_xticklabels(), rotation=90)

ax[0].set_title("Average no. of in-app purchases game genre wise")

ax[1].set_title("Average total value of in-app purchases game genre wise")

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(df[["Price","Average User Rating","Total In-app Purchases","Age Rating","Size"]].corr(), annot=True, cmap="coolwarm")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(20,6))

df.groupby("Release Year")["Name"].agg("count").plot(ax=ax[0])

df.groupby("Release Month")["Name"].agg("count").plot(ax=ax[1])

ax[0].set_ylabel("No. of apps")

ax[1].set_ylabel("No. of apps")

ax[0].set_title("No. of apps released in each year")

ax[1].set_title('No. of apps released in each month')

plt.show()