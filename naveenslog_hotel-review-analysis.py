# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in kernelb9a3c46a04



import nltk

import eli5

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Hotel_Reviews.csv")

df.head(3)
df.describe()
df.isna().sum()
import reverse_geocoder as rg



def get_geo_data(x):

    try:

        lat = x["lat"]

        lng = x["lng"]



        coordinates = (lat,lng)

        results = rg.search(coordinates) # default mode = 2

        return results

    except Exception as e:

        print(e)

        return





unique_hotel_address = df[["Hotel_Address","lat", "lng"]].drop_duplicates()

unique_hotel_address["geo_details"] = unique_hotel_address.apply(get_geo_data, axis=1)

df = pd.merge(df,unique_hotel_address[["Hotel_Address", "geo_details"]], on="Hotel_Address", how="left")

df = df[df['geo_details'].isna() == False]



## Fetching city and country 

df["cc"] = df["geo_details"].apply(lambda x: x[0]["cc"])

df["city"] = df["geo_details"].apply(lambda x: x[0]["name"])
## Plotting country wise average review score

plt.subplots(figsize=(15,5))

sns.boxplot(x="cc", y="Average_Score", data=df)

plt.show()
## top 5 and bottom 5 hotel

bottom_5_hotels = (

    df.groupby(["Hotel_Name"])["Average_Score"].mean()

                                               .sort_values()

                                               .reset_index()

                                               .head()

)

top_5_hotels = (

    df.groupby(["Hotel_Name"])["Average_Score"].mean()

                                               .sort_values(ascending=False)

                                               .reset_index()

                                               .head()

)
plt.subplots(figsize=(15,5))

sns.barplot(x="Hotel_Name", y="Average_Score", data=bottom_5_hotels, color="blue")

plt.show()
plt.subplots(figsize=(15,5))

sns.barplot(x="Hotel_Name", y="Average_Score", data=top_5_hotels, color="blue")

plt.show()