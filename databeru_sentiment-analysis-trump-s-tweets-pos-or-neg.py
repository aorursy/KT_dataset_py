import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/trump-tweets/realdonaldtrump.csv")
df.head(5)
# Select only the columns, which will be used
df = df.drop(["id","link"], axis = 1)

df.describe()
df.info()
plt.figure(figsize=(8,6))
sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 15)
plt.show()
df["date"] = pd.to_datetime(df["date"])
df["date"].apply(lambda x: x.year)

# Number of tweets by year
colors = []
for i in range(2020-2009+1):
    x = 0.7-0.06*i
    c = (x,x,0.5)
    colors.append(c)

bar = df["date"].apply(lambda x: x.year).value_counts().sort_index().plot.bar(figsize = (16,10), color = colors)
plt.title("Number of tweets by year\n", fontsize=20)
bar.tick_params(labelsize=14)
plt.axvline(8, 0 ,1, color = "grey", lw = 3)
plt.text(7.7, 8800, "President", fontsize = 18, color = "grey")
bar.tick_params(labelsize=18)
plt.show()

# Number of tweets (more details)
df["year_month"] = df["date"].apply(lambda x: str(x.year)+"-"+str(x.month))
df["year_month"] = pd.to_datetime(df["year_month"])
year_month = pd.pivot_table(df, values = "content", index = "year_month", aggfunc = "count")

bar = year_month.plot(figsize = (16,10))
plt.title("Number of tweets (more details)", fontsize=20)
plt.axvline(8, 0 ,1, color = "grey", lw = 3)
bar.tick_params(labelsize=18)
plt.legend("")
plt.xlabel("")
bar.get_yaxis().set_visible(False)
plt.show()
# Calculate the polarity of the tweets of Trump with NLTK
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

df["polarity"] = df["content"].apply(lambda x: sentiment.polarity_scores(x))

df["pos"] = df["polarity"].apply(lambda x: x["pos"])
df["neg"] = df["polarity"].apply(lambda x: x["neg"])
df["compound"] = df["polarity"].apply(lambda x: x["compound"])

# Create the visualization
fig = plt.figure(figsize = (14,10))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Mean positivity/negativity of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Positivity plot
year_month = pd.pivot_table(df, values = "pos", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Negativity plot
year_month = pd.pivot_table(df, values = "neg", index = "year_month", aggfunc = "mean").apply(lambda x: -x)
ax.plot(year_month, lw = 5, color = "red")

# Add the "president" and "corona" lines
ax.legend(["pos","neg"], fontsize=18)
plt.axhline(0, 0 ,1, color = "black", lw = 1)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -0.18, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 0.39, "Corona", fontsize = 18, color = "orange")
ax.tick_params(labelsize=18)
plt.show()
# Create the visualization
fig = plt.figure(figsize = (14,6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Compound positivity/negativity of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Compound plot
df["year_month"] = df["date"].apply(lambda x: str(x.year)+"-"+str(x.month))
df["year_month"] = pd.to_datetime(df["year_month"])
year_month = pd.pivot_table(df, values = "compound", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5, color = "green")


# Add the "president" and "corona" lines
ax.legend(["pos","neg"], fontsize=18)
plt.axhline(0, 0 ,1, color = "black", lw = 1)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -0.18, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 0.7, "Corona", fontsize = 18, color = "orange")
plt.legend("")
plt.show()

# Create the visualization
fig = plt.figure(figsize = (14,6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Quantity of retweets and favorites of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Monthly Average number of "retweets"
year_month = pd.pivot_table(df, values = "retweets", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Monthly Average number of "favorites"
year_month = pd.pivot_table(df, values = "favorites", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Add the "president" and "corona" lines
ax.legend(["Retweets","Favorites"], fontsize=18)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -13000, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 140000, "Corona", fontsize = 18, color = "orange")
ax.tick_params(labelsize=18)
plt.show()

plt.figure(figsize = (10,8))
sns.heatmap(df[["retweets","favorites", "compound"]].corr(), annot = True, cmap="YlGnBu")
plt.title("Correlation between retweets, favorites and compound\n", fontsize = 14)
plt.show()
plt.figure(figsize=(10,8))
sns.scatterplot("compound","retweets", data = df, alpha = 0.5)
plt.title("Relation between retweets and compound (positivity/neg.)", fontsize = 15)
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot("compound","retweets", data = df, alpha = 0.01)
plt.title("Relation between retweets and compound (positivity/neg.)", fontsize = 15)
plt.show()