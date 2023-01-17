import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")

df.head()

df.drop("Unnamed: 0", axis = 1, inplace = True)
pd.set_option('display.max_columns', None)

df.head()
def clean_name(x):

    return x.lower().strip().replace(" ", "_")

df = df.rename(columns = clean_name)
# Take a look at top 10 movies according to IMDB Rating 

sns.set()

plt.figure(figsize = (10,8))

plt.title("Top 20 Movies according to IMDB rating")

top_20_imdb = df.sort_values(by = "imdb", ascending = False)[["title", "imdb"]][:20]

sns.barplot(data = top_20_imdb, y = "title", x="imdb", palette="ocean")

plt.xlabel("Ratings")

plt.ylabel("Movies")

plt.show()
# Clean column Age 

df.isna().sum()
df.dtypes
df.age.value_counts(dropna=False)

df.age = df.age.str.replace("+","").replace("all", 0).astype(float)
# filling NAN values in age  with the mean age



# filling NAN values in imdb with mean rating 



# filling NAN values in runtime with mean time



df[["age", "imdb", "runtime"]] = df[["age", "imdb", "runtime"]].fillna(df[["age", "imdb", "runtime"]].mean())
# Track rating year wise

plt.title("Average rating trend over the course of time", size = 25)

_ = df.groupby("year")["imdb"].mean().plot(figsize = (15,6))

_ = plt.xlabel("Year")

_ = plt.ylabel("Rating")
# Produced movie counts of last 17 years

sns.set()

plt.figure(figsize = (20,8))

plt.title("No of movies produced in recent years", size = 15)

movie_count = df.year.value_counts()[:20].reset_index().rename(columns = {"index": "year" , "year" : "count"})

sns.barplot(data = movie_count, x = "year", y ="count", palette="winter")

plt.xlabel("Years", size = 15)

plt.ylabel("count of movies", )

plt.show()
# avergae age of watching for movies 

sns.set()

plt.title("Average watching year", size = 18)

_ = df.groupby("year")["age"].mean().plot(figsize = (15,6))

_ = plt.xlabel("Year", size = 15)

_ = plt.ylabel("Age", size = 15)
# Taking languistically

languistic = df.language.value_counts()[:5].reset_index().rename(columns = { "index": "language", "language" : "count"}) 
# Produced movie counts of last 17 years

sns.set()

plt.figure(figsize = (20,8))

plt.title("Languages of movies", size = 15)

sns.barplot(data = languistic, x = "count", y ="language", palette="winter")

plt.xlabel("count", size = 15)

plt.ylabel("Languages  for movies", )

plt.show()
sns.set()

plt.figure(figsize = (10,10))

sns.scatterplot(data =df, y="imdb", x="runtime", palette="inferno")

plt.show()
sns.set()

plt.figure(figsize = (10,10))

sns.relplot(data =df, y="imdb", x="runtime", col = "netflix", palette="inferno", hue = "netflix", kind = "scatter")

plt.show()