import pandas as pd
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/UserRatings1.csv")
print(df.shape)
df.head()
df = df.melt(id_vars="JokeId",value_name="rating")
df.dropna(inplace=True)
df.variable = df.variable.str.replace("User","")
df.rename(columns={"variable":"User"},inplace=True)
print(df.shape)
df.head()
df["mean_joke_rating"] = df.groupby("JokeId")["rating"].transform("mean")
df["mean_user_rating"] = df.groupby("User")["rating"].transform("mean")
df["user-count"] = df.groupby("User")["rating"].transform("count")
df["joke-count"] = df.groupby("JokeId")["rating"].transform("count")
df.describe()
df.head()
df = df.merge(pd.read_csv("../input/JokeText.csv"),on="JokeId")
df.tail()
df.rating.mean()
df.rating.median()
df3 = df.drop_duplicates(subset=["JokeId"]).loc[:,['JokeId', 'mean_joke_rating', 'joke-count', 'JokeText']].sort_values('mean_joke_rating',ascending=False)
df3.head()

for j in list(df3.head().JokeText): print (j)
# The least funny?
for j in list(df3.tail().JokeText): print (j)

df.to_csv("jokerRatingsMerged.csv.gz",index=False,compression="gzip")