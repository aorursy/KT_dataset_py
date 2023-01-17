import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
# Load the  NoteBook

df = pd.read_csv("/kaggle/input/restaurants-in-bangalore/CleanedBangaloreHotels.csv")
df.head()
df.dtypes
temp = df[["price", "ratings"]]
temp[temp.price.str.contains("\n")]
sample = temp.price.str.replace("<bound method Series.mean of 1 100\n2 ...", 'NA')
sample.iloc[sample[sample.str.contains("\n")].index] = np.nan
df.price = sample.astype(float)
temp.ratings = temp.ratings.str[:3]
df.ratings = temp.ratings.astype(float)
df.dtypes
df.head()
df.cuisine.nunique ()
top_cuisine = df.cuisine.value_counts().reset_index().sort_values("cuisine", ascending = False)[:5]
%matplotlib inline

sns.set(font_scale = 1.5)

plt.figure(figsize = (15,6))

sns.barplot(x=top_cuisine["index"], y = top_cuisine["cuisine"], palette="coolwarm")

plt.show()
mean_price = []

for i in top_cuisine["index"].to_list():

    avg_price = df[df["cuisine"] == i]["price"].mean()

    mean_price.append(avg_price)

mean_price_df = pd.DataFrame(top_cuisine["index"])

mean_price_df["avg_price"] = mean_price

mean_price_df.rename(columns = {"index": "cuisine"}, inplace = True)
sns.set(font_scale = 1.5)

plt.figure(figsize = (8,6))

plt.title("Average Price of each most occuring food type in banglore", fontdict={'fontsize':20})

sns.barplot(x=mean_price_df["avg_price"], y = mean_price_df["cuisine"], palette="inferno")

plt.show()
ratings_view = df.ratings.value_counts().reset_index().sort_values(by= "index", ascending = False)
sns.set()

plt.figure(figsize = (15,8))

plt.title("Distribution of the rating for resto in banglore", fontdict={'fontsize':20})

sns.barplot(x=ratings_view["index"], y = ratings_view["ratings"], palette="inferno")

plt.show()
sns.set(font_scale = 1.5)

plt.figure(figsize = (25,8))

plt.title("Distribution of the Price with rating for resto in banglore", fontdict={'fontsize':20})

sns.lineplot(y=df["price"], x = df["ratings"])

plt.show()
top_resto = pd.DataFrame()

new_temp = pd.DataFrame()

for i in mean_price_df["cuisine"]:

    temp = df[df["cuisine"] == i]

    print("Top restaurants for " , i.upper() , "(according to ratings)")

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    print(temp[temp['ratings'] > 4.0]["name"][:5])

    print("-------------------------------------")
plt.figure(figsize = (8,8))

sns.set(font_scale = 1.5)

plt.title("Most occuring food tags in banglore", fontdict={'fontsize':30})

sns.barplot(y="index", x="tags" , data = df.tags.value_counts()[:10].reset_index())

plt.show()
sns.set(font_scale = 1.2)

plt.figure(figsize = (8,8))

sns.scatterplot(y="price", x="ratings", data = df, hue=df["ratings"],size=df["ratings"] , palette="rainbow")

plt.show()