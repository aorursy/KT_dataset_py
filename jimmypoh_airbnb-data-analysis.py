# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



airbnb = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
airbnb.describe()
airbnb.info()
airbnb.head()
# I will drop columns "host_id" and "host_name" as I do not need them in data analysis

airbnb.drop(["host_id","host_name","last_review"], axis=1, inplace=True)
airbnb.isnull().sum()

# there are some missing values
# I will fill "review_per_month" missing values with 0 

airbnb["reviews_per_month"] = airbnb["reviews_per_month"].fillna(0)
# as for column "name" missing values, I choose not to drop them as I might discard important information

# I decided to fill them with "no comment" 

airbnb["name"] = airbnb["name"].fillna("no comment")
airbnb.isnull().sum()

# all missing values are filled
airbnb["neighbourhood_group"].unique()
airbnb["neighbourhood"].unique()
airbnb["room_type"].unique()
############################# Exploratory Data Analysis ###############################

plt.figure(figsize=(16,8))

sns.countplot(x="room_type", data=airbnb)

plt.show()

# it shows that entire house/apt is the most popular choice among other room types
plt.figure(figsize=(16,8))

sns.countplot(x="neighbourhood_group", hue="room_type", data=airbnb)

plt.show()

# it seems that Manhattan has highest entire home/apt among other neighbourhood groups

# perhaps majority of guests go to Manhattan are either group of friends or family members

# shared room has very significant low number among all neighbourhood groups

# possibly couples would not choose those neighbourhood groups as their vacation spots
df_price = airbnb.groupby("neighbourhood_group").agg({"price":"mean"}).sort_values("price", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="neighbourhood_group", y="price", data=df_price)

plt.show()

# it seems Manhattan charges highest average price among other neighbourhood groups

# perhaps Manhattan is the most popular vacation spot

# this is also due to Manhattan has the highest number of entire home/apt among other neighbourhood groups

# entire home/apt will charge higher price than private and shared rooms
df_night = airbnb.groupby("neighbourhood_group").agg({"minimum_nights":"mean"}).sort_values("minimum_nights", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="neighbourhood_group", y="minimum_nights", data=df_night)

plt.show()

# it seems most of guests would like to stay at Manhattan longer than other neighbourhood groups
df_review = airbnb.groupby("neighbourhood_group").agg({"number_of_reviews":"sum"}).sort_values("number_of_reviews", ascending=False).reset_index()



plt.figure(figsize=(16,8))

sns.barplot(x="neighbourhood_group", y="number_of_reviews", data=df_review)

plt.show()

# it seems that Brooklyn has the highest sum of number of reviews among other neighbourhood groups

# but we cannot tell from here whether those reviews are good or bad

# what we can tell from here is guests are defenitely love to review about Brooklyn accomodations
df_neighbourhood = airbnb["neighbourhood"].value_counts().head(10).reset_index()

df_neighbourhood.columns = ["neighbourhood","counts"]



plt.figure(figsize=(16,8))

sns.barplot(x="neighbourhood", y="counts", data=df_neighbourhood)

plt.show()

# it seems Williamsburg is the most popular choice for guests to stay overnight

# Williamsburg is a hip neighborhood in Brooklyn and it is one of the popular vacation spot.
import folium

from folium.plugins import HeatMap



airbnb_map = folium.Map(location = [40.71,-74.01], zoom_start=11)

HeatMap(airbnb[["latitude","longitude"]],radius=8, gradient={0.4:"blue",0.65:"purple",1.0:"red"}).add_to(airbnb_map)

airbnb_map

# it seems like center of New York is highly populated.
# I will try to create a wordcloud on column "name"

# Before that, I will clean up the data first

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



stop_words = stopwords.words("english")

lem = WordNetLemmatizer()



def cleaning_words(i):

    i = re.sub("[^A-Za-z]+"," ", i).lower()

    i = re.sub("[0-9]+", " ",i)

    lem_words = []

    for k in i.split(" "):

        k = lem.lemmatize(k)

        lem_words.append(k)

    words = []

    for x in lem_words:

        if x not in stop_words:

            words.append(x)

    w = []

    for z in words:

        if len(words) > 3:

            w.append(z)

    return(" ".join(w))            
wordnet = airbnb["name"].apply(cleaning_words)

wordnet = " ".join(wordnet)
from wordcloud import WordCloud



wordcloud_airbnb = WordCloud(background_color="black",height=1400, width=1800).generate(wordnet)

plt.figure(figsize=(16,8))

plt.imshow(wordcloud_airbnb)

# quite lots of guests mentioned about "Williamsburg", "Brooklyn", "private room", "private bedroom"

# perhaps there are high number of solo traveller or backpacker travelling to Brooklyn since Brooklyn provides the highest number of private room among other neighbourhood groups.
# As a conclusion, hosts at Brooklyn can consider to provide more supports and facilities specifically for solo traveller or backpacker.

# This may bring more business income for hosts at Brooklyn.