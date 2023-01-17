import matplotlib.pyplot as plt #for visualisaton

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing data

imdb_data = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv")



#getting overview of various columns

imdb_data.info()
#finding dimensions

print(imdb_data.shape)
#Let's see how many columns contains NA values

imdb_data.isna().any()
#Getting count of NA values in each column

print(imdb_data.isna().sum())



#visualizing

imdb_data.isna().sum().plot(kind="bar")

imdb_data_cleaned = imdb_data.dropna()

imdb_data_cleaned.info()
#finding summary statistics 

imdb_data_cleaned.describe()

#visulizing the histogram of ratings

imdb_data_cleaned["Rating"].hist(bins=30)
revenue_hist = imdb_data_cleaned["Revenue (Millions)"].hist(bins=30)

revenue_hist.set_xlabel("Revenue (in Million $)")

revenue_hist.set_ylabel("Movie Counts")
revenue_hist_zoomed = imdb_data_cleaned["Revenue (Millions)"].hist(bins=30)

revenue_hist_zoomed.set_xlim(0, 200)

revenue_hist_zoomed.set_xlabel("Revenue (in Million $)")

revenue_hist_zoomed.set_ylabel("Movie Counts")
imdb_data_cleaned.plot(kind="scatter", x="Rating", y="Revenue (Millions)", color="orchid")
imdb_data_cleaned.plot(kind="scatter", x="Rating", y="Revenue (Millions)", color="orange", ylim=(0, 500), alpha=0.4)
top_rated = imdb_data_cleaned.sort_values(["Rating","Metascore"], ascending=False)[

    ["Title", "Director", "Rating","Metascore"]]

top_rated.index = range(1,839)

top_rated.head(n=15)
#Caution: MultiIndex Dataframe

top_rated.groupby("Director")[["Rating", "Metascore"]].agg([np.mean, np.median]).sort_values(

    [("Rating","mean"),("Metascore", "mean")], ascending = False).head(n=15)
top_rated_revenue = imdb_data_cleaned.sort_values(["Rating","Metascore"], ascending=False)[

    ["Title", "Director", "Rating","Metascore", "Revenue (Millions)"]]

top_rated_revenue.index = range(1,839)

top_rated_revenue.head(n=15)
top_rated_runtime = imdb_data_cleaned.sort_values(["Rating", "Metascore"], ascending=False)[

    ["Title", "Director", "Runtime (Minutes)", "Rating","Metascore"]]

top_rated_runtime.index = range(1,839)

top_rated_runtime.head(n=15)
#to see if there is any correlatiob between runtime and metascore

imdb_data_cleaned[["Runtime (Minutes)", "Metascore"]].corr()
#Let's plot with respect to Metascore because, it is more unique

top_rated_runtime.plot(kind="scatter",

                      x="Runtime (Minutes)",

                      y="Metascore",

                      alpha=0.4)
imdb_data_cleaned[["Votes", "Metascore"]].corr()
imdb_data_cleaned.plot(kind="scatter",

                      x="Votes",

                      y="Metascore",

                      color="red",

                      alpha=0.4,

                      )
#zooming to clustered area

imdb_data_cleaned.plot(kind="scatter",

                      x="Votes",

                      y="Metascore",

                      color="red",

                      alpha=0.4,

                      xlim=(0, 650000)

                      )
year_vs_revenue = imdb_data_cleaned.groupby("Year")[["Revenue (Millions)"]].mean()

year_vs_revenue.plot(kind="bar", color="green")