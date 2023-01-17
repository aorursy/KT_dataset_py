# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
df = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv")
df.head()
df.describe().T
df.nlargest(15,"Rating")[["Title", "Director", "Rating", "Votes", "Revenue (Millions)"]].set_index("Title")
df.nlargest(15, "Votes")[["Title", "Director", "Rating", "Votes", "Revenue (Millions)"]].set_index("Title")
df.nlargest(15, "Revenue (Millions)")[["Title", "Director", "Rating", "Votes", "Revenue (Millions)"]].set_index("Title")
df.groupby(["Year"])["Rating"].mean().plot(kind = "bar", figsize = (12,8), fontsize = 12, color = "yellow")
plt.title("Average IMDb Raiting By Years", fontsize = 13)
df.groupby(["Year"])["Metascore"].mean().plot(kind = "bar", figsize = (12,8), fontsize = 12, color = "green")
plt.title("Average Metascore By Years", fontsize = 13)
df.groupby(["Year"])["Revenue (Millions)"].mean().plot(kind = "bar", figsize = (12,8), fontsize = 12, color = "orange")
plt.title("Average Revenue By Years", fontsize = 13)
df.groupby(["Year"])["Runtime (Minutes)"].mean().plot(kind = "bar", figsize = (12,8), fontsize = 12, color = "lightblue")
plt.title("Average Runtime By Years", fontsize = 13)
bm = df.sort_values(by = ["Rating"], ascending = False).groupby("Year").first()
bm.groupby(["Title", "Year"])["Rating"].mean().sort_values().plot.bar(x = "Title", y = "Rating", fontsize = 14, figsize = (14,8), color = "pink")
plt.title("Highest Rated Movies by Years", fontsize = 13)
br = df.sort_values(by = ["Revenue (Millions)"], ascending = False).groupby("Year").first()
br.groupby(["Title", "Year"])["Revenue (Millions)"].mean().sort_values().plot.bar(x = "Title", y = "Revenue (Millions)", fontsize = 14, figsize = (14,8), color = "purple")
plt.title("Highest Revenue Movies by Years", fontsize = 13)
df.groupby(["Director"])["Revenue (Millions)"].mean().sort_values(ascending = False)
df.groupby(["Director"])["Revenue (Millions)"].mean().sort_values(ascending = False)[:15].plot(kind = "bar", figsize = (15,8), fontsize = 13, color = "salmon")
plt.title("Average Revenue of Directors")
df.groupby(["Director"])["Rating"].mean().sort_values(ascending = False)
df.groupby(["Director"])["Rating"].mean().sort_values(ascending = False)[:20].plot(kind = "bar", figsize = (16,8), fontsize = 13, color = "goldenrod")
plt.title("IMDb Average of Directors")
df.groupby(["Director"])["Metascore"].mean().sort_values(ascending = False)[:20].plot(kind = "bar", figsize = (16,8), fontsize = 13, color = "mediumpurple")
plt.title("Metascore Average of Directors")
top_genres_revenue = df.groupby(['Genre']).mean().sort_values(by = "Revenue (Millions)", ascending = False)[:20]
print(top_genres_revenue[["Rank", "Revenue (Millions)", "Rating"]])
plt.figure(figsize = (16,12))
plt.title("Genre Groups with Highiest Average Revenue", fontsize = 13)
sns.barplot(top_genres_revenue["Revenue (Millions)"], top_genres_revenue.index, palette = "BuPu")
plt.show()
top_genres_imdb = df.groupby(["Genre"]).mean().sort_values(by = "Rating", ascending = False)[:20]
print(top_genres_imdb[["Rank", "Revenue (Millions)", "Rating"]])
plt.figure(figsize = (16,12))
plt.title("Genre Groups with Highiest Average IMDb", fontsize = 13)
sns.barplot(top_genres_imdb["Rating"], top_genres_imdb.index, palette = "GnBu")
plt.show()
genres = list(genre.split(',') for genre in df.Genre)
genres = list(itertools.chain.from_iterable(genres))
genres = pd.value_counts(genres)
print(genres)
plt.figure(figsize = (16,12))
plt.title("Most Movie Types", fontsize = 13)
sns.barplot(genres.values, genres.index, palette = sns.color_palette("coolwarm", 20))
plt.show()
actors = list(actor.split(',') for actor in df.Actors)
actors = list(itertools.chain.from_iterable(actors))
actors = pd.value_counts(actors)
print(actors)
plt.subplots(figsize = (12, 8))
sns.heatmap(df.corr(), annot = True, cmap = "PuBu")
plt.title("IMDb Correlation", fontsize = 13)
plt.show()
sns.set_style("darkgrid")
sns.lmplot(data= df, y = "Metascore", x = "Rating", height = 10)
plt.title("IMDb Point vs Metascore", fontsize = 13)
plt.show()
sns.set_style("darkgrid")
sns.lmplot(data= df, y = "Votes", x = "Rating", height = 10)
plt.title("IMDb Point vs Votes", fontsize = 13)
plt.show()
sns.set_style("darkgrid")
sns.lmplot(data= df, y = "Runtime (Minutes)", x = "Rating", height = 10)
plt.title("IMDb Point vs Runtime", fontsize = 13)
plt.show()
sns.set_style("darkgrid")
sns.lmplot(data= df, y = "Revenue (Millions)", x = "Rating", height = 10)
plt.title("IMDb Point vs Revenue", fontsize = 13)
plt.show()