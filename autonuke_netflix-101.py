import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline 



sns.set(style="darkgrid")



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/Netflix Shows.csv', encoding='cp437')

df.head()
df["title len"] = [len(i.split()) for i in df['title']]



dates_only = df[['release year']]

grouped_mean = df.groupby("release year")["title len"].mean().reset_index().sort_values(by='release year',ascending=False).reset_index(drop=True)

grouped_max = df.groupby("release year")["title len"].max().reset_index().sort_values(by='release year',ascending=False).reset_index(drop=True)

grouped_mean['agg_type'] = "mean"

grouped_max['agg_type'] = "max"

grouped = df[["release year", 'title len']]



plt.figure(figsize=(12,6))



grouped = grouped_mean.append(grouped_max)







sns.barplot(x='release year',y='title len', data=grouped, palette="muted", hue="agg_type")

plt.xticks(rotation=75)
grouped_rating = df.groupby(["release year", 'rating'])["title len"].mean().reset_index().sort_values(by='release year',ascending=False).reset_index(drop=True)

#grouped_rating = grouped_rating[(grouped_rating["release year"] > 1999)]

pivot = grouped_rating.pivot("rating", "release year", "title len")

# pivot = pivot[(pivot["release year"] > 2000)]

fig, ax = plt.subplots(figsize=(18,10)) 

sns.heatmap(pivot, annot=True, fmt=".1f", linewidths=.9, ax=ax)

plt.xticks(rotation=75)
grouped_rating = df.groupby("rating")["title len"].mean().reset_index().sort_values(by='rating',ascending=False).reset_index(drop=True)

sns.barplot(x='rating',y='title len', data=grouped_rating, color="salmon")

plt.xticks(rotation=75)