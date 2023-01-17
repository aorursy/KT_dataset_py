import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

ab=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#Checking the most present neighbourhood group

sns.catplot(x="neighbourhood_group", kind = "count", palette = "Set2", data = ab)

plt.show()
## Now checking the most present neighbourhood 



neighbourhood_top10 = ab["neighbourhood"].value_counts().head(10)

df_neighbourhood_top10 = pd.DataFrame(neighbourhood_top10)

df_neighbourhood_top10 = df_neighbourhood_top10.reset_index()

f, ax = plt.subplots(figsize = (15,5))

sns.barplot(x ="index", y = "neighbourhood", palette = "Set2", data = df_neighbourhood_top10)

plt.show()
sns.catplot(x="neighbourhood_group", y="price", kind = "box",data = ab)

plt.show()
ab_no_outliers = ab[["neighbourhood_group","price"]]



#Here we're going to show only the prices below 400. We have 48895 rows with the full dataset, by doing this filter we will

#have 45338 rows, which means a lost of 7% of the rows due the filter.



ab_no_outliers_filteres = ab_no_outliers[ab_no_outliers["price"] <= 400]



sns.catplot(x="neighbourhood_group", y="price", kind = "box", data = ab_no_outliers_filteres)

plt.show()
ab_price = ab.groupby(["room_type"])["price"].median()

df_ab_price = pd.DataFrame(ab_price)

df_ab_price = df_ab_price.reset_index()



sns.catplot(x="room_type", y="price", kind = "bar", palette = "Accent",  data = df_ab_price)

plt.title("Room_type price by it's median")

plt.show()
ab_room = ab[["neighbourhood_group","room_type","price"]]



ab_room_no_outliers = ab_room[ab_room["price"]<=300]



ax = sns.catplot(x = "neighbourhood_group", y = "price", kind = "box", hue = "room_type", data = ab_room_no_outliers)





plt.show()
ab_reviews = ab.groupby(["neighbourhood_group"])["number_of_reviews"].sum()

df_ab_reviews = pd.DataFrame(ab_reviews)

df_ab_reviews = df_ab_reviews.reset_index()



sns.barplot(x="neighbourhood_group", y="number_of_reviews", data = df_ab_reviews)

plt.title("Total Reviews by neighbourhood_group")

plt.show()
sns.relplot(x="latitude", y="longitude", palette = "Set2", hue = "neighbourhood_group", data = ab)

plt.show()
ab_price = ab[["longitude", "latitude", "price", "neighbourhood_group","room_type"]]

ab_price_under_100 = ab_price[ab_price["price"]<=100]



sns.relplot(x="latitude", 

            y="longitude", 

            palette = "Set2", 

            hue = "neighbourhood_group", 

            style = "room_type", 

            data = ab_price_under_100)

plt.title("Room_type under $100")

plt.axis("off")

plt.show()
ab_price = ab[["longitude", "latitude", "price", "neighbourhood_group","room_type"]]

ab_price_above_500 = ab_price[ab_price["price"]>=500]



sns.relplot(x="latitude", 

            y="longitude", 

            palette = "Set2", 

            hue = "neighbourhood_group", 

            style = "room_type", 

            data = ab_price_above_500)

plt.title("Room_type above $500")

plt.axis("off")

plt.show()
ab_night = ab.groupby(["neighbourhood_group"])["minimum_nights"].mean().round(2)

df_ab_night = pd.DataFrame(ab_night)

df_ab_night = df_ab_night.reset_index()



sns.catplot(x="minimum_nights", y = "neighbourhood_group", kind = "bar", data = df_ab_night)

plt.title("Minimum_nights mean by neighbourhood_group")

plt.show()
ab_proportion = ab.groupby(["neighbourhood_group"])["room_type"].value_counts()

df_ab_proportion = pd.DataFrame(ab_proportion)

df_ab_proportion.rename(columns={"room_type":"Total of values"}, inplace = True)





ab_count = ab.groupby(["neighbourhood_group"])["room_type"].count()

df_ab_count = pd.DataFrame(ab_count)





df_ab_proportion["Total"] = 0



df_ab_proportion.loc["Bronx"]["Total"]= df_ab_count.room_type.loc["Bronx"]

df_ab_proportion.loc["Brooklyn"]["Total"]= df_ab_count.room_type.loc["Brooklyn"]

df_ab_proportion.loc["Manhattan"]["Total"]= df_ab_count.room_type.loc["Manhattan"]

df_ab_proportion.loc["Queens"]["Total"]= df_ab_count.room_type.loc["Queens"]

df_ab_proportion.loc["Staten Island"]["Total"]= df_ab_count.room_type.loc["Staten Island"]



df_ab_proportion = df_ab_proportion.reset_index()



df_ab_proportion["Proportion"] = (df_ab_proportion["Total of values"]/df_ab_proportion["Total"]).round(2)



sns.catplot(x="neighbourhood_group",

            y = "Proportion",

            kind = "bar",

            hue = "room_type",

            data = df_ab_proportion)

plt.title("Room_type proportion for each neighbourhood_group")

plt.show()