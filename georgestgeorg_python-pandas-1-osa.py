import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
df = pd.read_csv("../input/movies.csv")
df
df.info()
len(df)
df["title"]
df["release_date"]
print("Keskmine eelarve:", df["budget"].mean())

print("Maksimaalne eelarve:", df["budget"].max())

print("Minimaalne eelarve:", df["budget"].min())

df["vote_average"].describe()
df["genres"].value_counts()
profit = df["revenue"] - df["budget"]

profit
(profit / 100000).round(1)
df["title"].str.upper()
release_date = pd.to_datetime(df["release_date"])

release_date.dt.year
pd.DataFrame({"pealkiri" : df["title"],

             "tulu (miljonites dollarites)" : (profit / 100000).round(1)})
df = df.assign(profit_in_millions=(profit / 100000).round(1),

              release_date=pd.to_datetime(df.release_date),

              release_year=pd.to_datetime(df.release_date).dt.year)

df
df[["title", "budget", "revenue", "profit_in_millions"]]
# välimised sulud on selleks, et saaks avaldise kahele reale kirjutada

(df[["title", "budget", "revenue", "profit_in_millions"]]

 .sort_values("profit_in_millions", ascending=False))
df["title"].sort_values()
df["title"].sort_values().head(3)
df["vote_average"] > 7.3
df[df["vote_average"] > 7.3]
df.title
df.groupby("release_year")["vote_average"].mean()
df.groupby(["release_year", "genres"])["vote_average", "vote_count"].mean()
df.groupby("release_year").aggregate({"vote_average": ["sum", "mean", "median"],

                                      "vote_count" : ["median", "sum"]})
df.vote_average.plot.hist(); # semikoolon väldib ebaolulise tekstilise kribukrabu kuvamist
df.vote_average.plot.hist(bins=11, grid=False, rwidth=0.95); 
df.plot.scatter("budget", "revenue", alpha=0.2);
df.to_csv("tulemus.csv")