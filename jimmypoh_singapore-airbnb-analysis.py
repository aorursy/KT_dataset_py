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



airbnb = pd.read_csv("/kaggle/input/singapore-airbnb/listings.csv")
airbnb.head()
# I will drop columns "id", "host_id", "host_name" as they are not necessary in analysis

airbnb = airbnb.drop(["id", "host_id", "host_name"], axis=1)
airbnb.describe()
airbnb.info()
airbnb.isnull().sum()

# there are some missing values
# I will drop rows containing missing values on column "name"

airbnb_index = airbnb["name"].dropna().index

airbnb = airbnb.iloc[airbnb_index,]
# as for "last_review" and "reviews_per_month" missing values, I choose to ignore for now as I want to do EDA on original data without disturbing its original distribution.
# I will change column "last_review" into datetime format

airbnb["last_review"] = pd.to_datetime(airbnb["last_review"])
# I will create a new column "year_of_review"

airbnb["year_of_review"] = airbnb["last_review"].dt.year
# I will create another new column "month_of_review"

airbnb["month_of_review"] = airbnb["last_review"].dt.month
airbnb.head()
################################## Exploratory Data Analysis #########################



plt.figure(figsize=(16,8))

sns.countplot("neighbourhood_group", data=airbnb)

plt.show()

# it seems that central region has highest number of hotels and accomodations.

# most likely central region is the most famous spot for vacation.
plt.figure(figsize=(16,8))

sns.countplot("neighbourhood_group", hue="room_type", data=airbnb)

plt.show()

# it seems that central region is also a famous vacation spot for family.

# as for regions other than central, they are preferred by solo traveller.
plt.figure(figsize=(16,8))

sns.violinplot(x="neighbourhood_group", y="price", data=airbnb)

plt.show()

# it seems there are some hotels or accomodations that are charging ridiculously high prices.

# I need to have a look on data to check whether they are due to error or other reason.
airbnb[airbnb["price"] > 6000]

# I will just check with those hotels that charge more than 6000

# from what I observe, those hotels that charge rediculously high prices are mostly luxury units.

# perhaps they offer the best facilities and luxury services for customers.

# however, judging from here, I cannot tell whether those luxury hotels are really up to what they are charging.
plt.figure(figsize=(16,8))

sns.violinplot(x="neighbourhood_group", hue="room_type", y="price", data=airbnb)

plt.show()

# surpringly, west region is the one that charges highest prices for private room and entire hoom/apt despite there are low number of rooms available at that region.
plt.figure(figsize=(16,8))

sns.violinplot(x="neighbourhood_group", y="minimum_nights", data=airbnb)

plt.show()

# central region has the highest minimum nights stayed.
plt.figure(figsize=(16,8))

sns.violinplot(x="neighbourhood_group", y="minimum_nights", hue="room_type", data=airbnb)

plt.show()

# at central region, private room has the highest minimum nights 
airbnb[airbnb["minimum_nights"] == 1000]

# from the information below, it is hard to explain why the following hotel has the highest minimum nights.

# I am not sure whether it is due to error or other reason.
plt.figure(figsize=(16,8))

sns.lineplot(x="year_of_review", y="price",hue="neighbourhood_group", data=airbnb)

plt.show()

# it seems that at year 2017, hotels and accomodations at west regions have the highest price.

# it seems that hotels at most regions have higher price at year 2014.

# maybe year 2014 and 2017 are popular for vacation at Singapore.
plt.figure(figsize=(16,8))

sns.countplot(x="neighbourhood_group",hue="room_type", data=airbnb[airbnb["year_of_review"] == 2017.0]);plt.title("Year 2017")

plt.show()

# it seems that central region has the highest number of stays despite west region has the highest price
plt.figure(figsize=(16,8))

sns.lineplot(x="month_of_review", y="price",hue="neighbourhood_group", data=airbnb)

plt.show()

# it seems that hotels at West region charges the highest price at around October.

# perhaps October is the good month for vacation at Singapore.

# I thought December will have the highest price.
import folium

from folium.plugins import HeatMap



airbnb_map = folium.Map(location = [1.29,103.85], zoom_start=12)

HeatMap(airbnb[["latitude","longitude"]], radius=8, gradient={0.4:"blue",0.65:"purple",1.0:"red"}).add_to(airbnb_map)

airbnb_map

# red indicates popular areas and most hotels and accomodations will be located at those areas
# As a conclusion, due to some missing values and outliers, there is a limit information I can extract out from this data.

# From what I can tell, it seems that central region is the most popular region for vacation.

# However, west region has hotels and accomodations that are charging the highest price.

# perhaps west region is suitable for ones who want to relax themselves, at the same time, enjoy luxury and outstanding services and facilities.