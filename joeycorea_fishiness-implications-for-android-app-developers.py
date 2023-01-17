import pandas as pd
import numpy as np
import time #is this required?
from re import sub
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

playstore = pd.read_csv("../input/googleplaystore.csv")
reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")

playstore.head(2)
reviews.head(2)
unique_values = playstore["Price"].unique()
unique_values.sort()
unique_values
everyone = playstore[playstore['Price'] == 'Everyone']
print(everyone)
broken = everyone.iloc[0]
app_name = broken["App"]
fixed = broken.shift(1)
fixed["App"] = app_name
#these value were found after finding the app on the live Playstore
fixed["Category"] = "Lifestyle"
fixed["Genres"] = "Lifestyle"
playstore.iloc[10472] = fixed
playstore.iloc[10472]
playstore[(playstore["Price"] == "0") & (playstore["Type"] != "Free")]
playstore.iloc[9148] = playstore.iloc[9148].set_value("Type", "Free")
playstore["Price"] = playstore["Price"].apply(lambda x: float(sub(r'[^\d\-.]', '', x)))
unique_values = playstore["Price"].unique()
unique_values.sort()
print(unique_values)
unique_values = playstore["Rating"].unique()
#unique_values.sort()
unique_values
nan_rating = playstore[playstore["Rating"].isnull()]
nan_rating.shape
nan_rating = playstore[(playstore["Rating" ].isnull()) & (playstore["Reviews"] != "0")]
nan_rating.shape
#create a loop that iterates through nan_rating and sets the viewed to 0 and pushes them back into playstore
nan_rating_idx = list(nan_rating.index.values)

for idx  in nan_rating_idx:
    playstore.iloc[idx] = playstore.iloc[idx].set_value("Rating", 0)

nan_rating = playstore[playstore["Rating"].isnull()]
print(nan_rating.shape)

playstore["Rating"] = playstore["Rating"].astype(float)
playstore["Price"] = playstore["Price"].astype(float)

plt.figure()
plt.scatter(playstore["Price"], playstore["Rating"])
plt.xlabel("Price ($)")
plt.ylabel("Rating (Stars)")
plt.show()
average_price = playstore["Price"].groupby(playstore["Rating"]).mean()
plt.figure()
plt.scatter(average_price.index.values, average_price.values)
plt.xlabel("Rating (Stars)")
plt.ylabel("Average Price ($)")
plt.show()
average_price.head(5)
average_price.tail(5)
average_rating = playstore["Rating"].groupby(playstore["Price"]).mean()
average_rating = pd.DataFrame(data=average_rating)
average_rating = average_rating.reset_index()
average_rating.loc[(average_rating["Price"] > 50) & (average_rating["Rating"] > 3.5), "Expensive_App_Highly_Rated"] = 1
average_rating["Expensive_App_Highly_Rated"].fillna(0, inplace=True)
plt.figure(figsize=(25,12))
#sns.scatterplot(average_rating.index.values, average_rating.values, hue="Expensive_App_Highly_Rated", data=df)
sns.scatterplot(x="Price", y="Rating", hue="Expensive_App_Highly_Rated", data=average_rating)
plt.xlabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.legend().remove()
plt.show()
expensive_apps = playstore[(playstore["Price"] > 50) & (playstore["Rating"] > 3.5)]
expensive_apps[["App", "Category", "Genres", "Price", "Rating", "Reviews", "Installs"]].sort_values("Price", ascending=False)
reviews.head(2)
for name, row in playstore.iterrows():
    genres_list = row["Genres"].split(";")
    for genre in genres_list:
        playstore.loc[name, str("gen_" + genre)] = 1
genre_dummies = [col for col in playstore if col.startswith('gen_')]
genre_dummies.sort()

aggregate_figures_per_genre_data = {}
for genre_dummy in genre_dummies:
    average_rating = playstore[playstore[genre_dummy] == 1]["Rating"].mean()
    average_price = playstore[playstore[genre_dummy] == 1]["Price"].mean()
    count = playstore[playstore[genre_dummy] == 1]["Rating"].count()
    aggregate_figures_per_genre_data[genre_dummy] = {
        "average_rating" : average_rating, 
        "average_price" : average_price, 
        "count" : count
    }
aggregate_figures_per_genre = pd.DataFrame(index=list(aggregate_figures_per_genre_data.keys()),data=list(aggregate_figures_per_genre_data.values()))
aggregate_figures_per_genre.head(5)
rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=aggregate_figures_per_genre.index.values, y=aggregate_figures_per_genre["average_rating"])
plt.xlabel("Genre", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.show()
rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=aggregate_figures_per_genre.index.values, y=aggregate_figures_per_genre["average_price"])
plt.xlabel("Genre", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Price ($)", fontsize=18)
plt.show()
average_rating_by_category = playstore["Rating"].groupby(playstore["Category"]).mean()
rating_plot = plt.figure(figsize=(25,12))
sns.barplot(x=average_rating_by_category.index.values, y=average_rating_by_category.values)
plt.xlabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.ylabel("Average Rating (Stars)", fontsize=18)
plt.show()
average_price_by_category = playstore["Price"].groupby(playstore["Category"]).mean()
price_plot = plt.figure(figsize=(25,12))
sns.barplot(x=average_price_by_category.index.values, y=average_price_by_category.values)
plt.ylabel("Price Point ($)", fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
plt.xlabel("Category", fontsize=18)
plt.show()