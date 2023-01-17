import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud

import re
df=pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()
df.shape
a=df.isnull().sum()

print(a.sum())
plt.figure(figsize=(16,8))

df["type"].value_counts().plot(kind="pie",shadow=True,autopct = '%1.1f%%')
count=list(df['country'].dropna().unique())

cloud=WordCloud(colormap="cool",width=800,height=400).generate(" ".join(count))

fig=plt.figure(figsize=(14,10))

plt.axis("off")

plt.imshow(cloud,interpolation=None)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["country"].value_counts()[:20].plot(kind="bar",color="lightcoral")

plt.title("Top 20 countries in terms of maximum number of movies on netflix",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["country"].value_counts()[:20].plot(kind="bar",color="blue")

plt.title("Top 20 countries in terms of maximum number of TV shows released on netflix",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(14,6))

df["release_year"].value_counts()[:20].plot(kind="bar",color="red")

plt.title("Frequency of TV shows and movies that are released in different years there in netflix",size=16)
plt.style.use("ggplot")

plt.figure(figsize=(14,6))

df[df["type"]=="Movie"]["release_year"].value_counts()[:20].plot(kind="bar",color="darkorange")

plt.title("Frequency of Movies which are released in different years and are there in netflix",size=16)
df[(df["type"]=="Movie") & (df["release_year"]==2017)]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(14,6))

df[df["type"]=="TV Show"]["release_year"].value_counts()[:20].plot(kind="bar",color="mediumblue")

plt.title("Frequency of TV shows which are released in different years and are there in netflix",size=16)
df[(df["type"]=="TV Show") & (df["release_year"]==2019)]["title"].sample(10)
listed=list(df['listed_in'].unique())

cloud=WordCloud(colormap="Wistia",width=600,height=400).generate(" ".join(listed))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for all category",size=18)
listed2=list(df[df["type"]=="Movie"]['listed_in'].unique())

cloud=WordCloud(colormap="summer",width=600,height=400).generate(" ".join(listed2))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for movie category",size=18)
listed3=list(df[df["type"]=="TV Show"]['listed_in'].unique())

cloud=WordCloud(colormap="YlOrRd",width=600,height=400).generate(" ".join(listed3))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for TV show category",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["listed_in"].value_counts()[:20].plot(kind="barh",color="red")

plt.title("20 most frequent genre for movie type for all the years",size=18)
df[(df["listed_in"]=="Documentaries") & (df["type"]=="Movie")]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["listed_in"].value_counts()[:20].plot(kind="barh",color="darkviolet")

plt.title("20 most frequent genre for TV show type for all the years",size=18)
df[(df["listed_in"]=="Kids' TV") & (df["type"]=="TV Show")]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["type"]=="Movie") & (df["release_year"]==2019)]["listed_in"].value_counts()[:20].plot(kind="barh",color="lime")

plt.title("20 most frequent Genre for movie type for the year 2019",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["type"]=="TV Show") & (df["release_year"]==2019)]["listed_in"].value_counts()[:20].plot(kind="barh",color="teal")

plt.title("20 most frequent genre for TV show type for the year 2019",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df["rating"].value_counts().plot(kind="bar",color="orange")

plt.title("Frequency of ratings for both TV shows & movies for all the years",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["rating"].value_counts().plot(kind="bar",color="royalblue")

plt.title("Frequency of ratings for movie category for all the years",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["rating"].value_counts().plot(kind="bar",color="orangered")

plt.title("Frequency of ratings for TV show category for all the categories",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="India") & (df["type"]=="Movie")]["rating"].value_counts().plot(kind="bar",color="red")

plt.title("Rating for Movies that are released in USA",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="India") & (df["type"]=="Movie")]["rating"].value_counts().plot(kind="bar",color="deeppink")

plt.title("Rating for Movies that are released in India",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="United States") & (df["type"]=="TV Show")]["rating"].value_counts().plot(kind="bar",color="fuchsia")

plt.title("Rating for TV Shows that are released in USA",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="India") & (df["type"]=="TV Show")]["rating"].value_counts().plot(kind="bar",color="gold")

plt.title("Rating for TV Shows that are released in India",size=18)
listed4=list(df[(df["release_year"]==2019) & (df["type"]=="Movie")]['title'])

cloud=WordCloud(colormap="YlOrRd",width=600,height=400).generate(" ".join(listed4))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for movie names which are released in the year 2019",size=18)
listed4=list(df[(df["release_year"]==2019) & (df["type"]=="TV Show")]['title'])

cloud=WordCloud(colormap="winter",width=600,height=400).generate(" ".join(listed4))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for TV Show category released in 2019",size=18)