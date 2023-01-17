# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import dataset
data = pd.read_csv("/kaggle/input/farmers-markets-in-the-united-states/wiki_county_info.csv")
data.head()
print(data.shape)
usa = pd.read_csv("/kaggle/input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv")
usa.head()
usa.shape
data.describe()
# Check the total null values
data.isnull().sum()
# county unique name
print(data["State"].nunique())
print(data["State"].duplicated().value_counts())
# drop the number column
data.drop("number", axis=1, inplace=True)
# fill county and state
data["county"].fillna("unknown", inplace=True)
data["State"].fillna("unknown", inplace=True)
print(usa.shape)
# farmers_markets_from_usda
usa.info()
data.head(7)
# import essential package
import matplotlib.pyplot as plt
import seaborn as sns
data.shape
# Group by State and per capita income
df = data.groupby("State")["per capita income"].sum().groupby("State").max().sort_values(ascending=False)
# Drop two row that contain null value
data.drop([2205, 2522],axis=0, inplace=True)
# Replace and convert as int

data["population"] = data["population"].str.replace(",","")
data["population"] = data["population"].astype(int)
data["number of households"] = data["number of households"].str.replace(",","")
data["number of households"] = data["number of households"].astype(int)
data["per capita income"] = data["per capita income"].str.replace("$","")
data["per capita income"] = data["per capita income"].str.replace(",","")
data["per capita income"] = data["per capita income"].astype(int)
data["median household income"] = data["median household income"].str.replace("$","")
data["median household income"] = data["median household income"].str.replace(",","")
data["median household income"] = data["median household income"].astype(int)
data["median family income"] = data["median family income"].str.replace("$","")
data["median family income"] = data["median family income"].str.replace(",","")
data["median family income"] = data["median family income"].astype(int)
# full information of dataset
data.info()
# State, county Vs per capita
# The term "per capita" is a Latin phrase that translates to "per person".

def state(state_name):
    data_state = data[data["State"]==state_name][["county", "per capita income", "population"]].sort_values("per capita income", ascending=False)
    return data_state
cap = state("Virginia")
df = pd.DataFrame(cap)
# visualization for virginia

sns.set(style="darkgrid", palette='Set1')
plt.subplots(figsize=(30,8))
plt.xticks(rotation=90)
sns.barplot(x=df["county"], y=df["per capita income"], data=df);
print("Per capita max: at {} {} min: at {} {}".format(df["county"][df["per capita income"].idxmax()], df["per capita income"].max(), df["county"][df["per capita income"].idxmin()], df["per capita income"].min()))
cap = state("New York")
df = pd.DataFrame(cap)
# visualization for New York

sns.set(style="darkgrid", palette='Set1')
plt.subplots(figsize=(30,8))
plt.xticks(rotation=90)
sns.barplot(x=df["county"], y=df["per capita income"], data=df);
print("Per capita max: at {} {} min: at {} {}".format(df["county"][df["per capita income"].idxmax()], df["per capita income"].max(), df["county"][df["per capita income"].idxmin()], df["per capita income"].min()))
# Now try to find out from every state which county has per capita income max
max_capita_county = data.groupby(["State", "county"])["per capita income"].sum().sort_values(ascending=False).reset_index()
max_capita_county
# create a new dataframe
df2 = pd.DataFrame(max_capita_county)
df2.head(20)
# Visualize first 20 per capita max value

plt.subplots(figsize=(16,5));
sns.barplot(x=df2[:20]["county"], y=df2[:20]["per capita income"], data=df2);
plt.xticks(rotation=90);
# print the county name and value
max_county = df2["county"][df2["per capita income"].idxmax()]
print("per capita income at {}".format(max_county))
print("per capita income max {}".format(df2["per capita income"].max()))
# Visualize last 20 per capita min value

plt.subplots(figsize=(16,5));
sns.barplot(x=df2[3210:]["county"], y=df2[3210:]["per capita income"], data=df2);
plt.xticks(rotation=90);
# print the county name and min value of all
max_county = df2["State"][df2["per capita income"].idxmin()]
print("per capita income at {}".format(max_county))
print("per capita income max {}".format(df2["per capita income"].min()))
data.head()
# population increase and number of households increase
data.plot(kind="scatter", x="population", y="number of households", figsize=(10, 5), alpha=0.5)
usa.head()
# Drop some columns
drop_list = ['FMID', 'street', 'zip', 'city', 'Season1Date', 'Season1Time', 'Season2Date',
'Season2Time', 'Season3Date', 'Season3Time', 'Season4Date', 'Season4Time', 'updateTime', 'Location']
usa.drop(drop_list, axis=1, inplace=True)
usa.columns
usa.head()
# Farmer market dataset information
usa.info()
# Farmers market data total null value
print(usa.isna().sum())
# Check unique value[not necessary]
print(usa["Organic"].unique())
# Replace the column Organic (-) value by (N)
usa["Organic"].replace("-", "N", inplace=True)
usa["Bakedgoods"].unique()
usa.head()
# Product list
food_df = ['Organic', 'Bakedgoods', 'Cheese', 'Crafts', 'Flowers', 'Eggs',
       'Seafood', 'Herbs', 'Vegetables', 'Honey', 'Jams', 'Maple', 'Meat',
       'Nursery', 'Nuts', 'Plants', 'Poultry', 'Prepared', 'Soap', 'Trees',
       'Wine', 'Coffee', 'Beans', 'Fruits', 'Grains', 'Juices', 'Mushrooms',
       'PetFood', 'Tofu', 'WildHarvested']

# Social platform and other media.
# for null value we assign 0, else 1
media = ['Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia']
for social in media:
    usa[social] = (usa[social].notnull().astype('int'))


# replace all the nan value by N
for i in food_df:
    usa[i].replace(np.nan, "N", inplace=True)


# import label encoder
from sklearn.preprocessing import LabelEncoder
# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()
for col in set(food_df):
    usa[col] = label_encoder.fit_transform(usa[col])
    
# payment method replace value by 1 and 0. If Y assign 1 else 0
payment_modes = ['Credit', 'WIC', 'WICcash', 'SFMNP', 'SNAP']
for payment_mode in payment_modes:
    try:
        usa[payment_mode] = usa[payment_mode].replace(to_replace=['Y', 'N'], value=[1,0])
    except:
        continue
# create new column
usa['payment type'] = usa.loc[:, 'WIC':'SNAP'].sum(1)
usa.head()
usa.columns
# create a dataframe for product
df_list = ['State' ,'County', 'Organic', 'Bakedgoods', 'Cheese', 'Crafts', 'Flowers', 'Eggs',
       'Seafood', 'Herbs', 'Vegetables', 'Honey', 'Jams', 'Maple', 'Meat',
       'Nursery', 'Nuts', 'Plants', 'Poultry', 'Prepared', 'Soap', 'Trees',
       'Wine', 'Coffee', 'Beans', 'Fruits', 'Grains', 'Juices', 'Mushrooms',
       'PetFood', 'Tofu', 'WildHarvested']

market = pd.DataFrame(usa, columns=df_list)
pd.set_option('display.max_columns', 50)
market.head()
# add all values from organic to wildharvested for different state and stored in new column called product
market["product"] = market.loc[:,'Organic':'WildHarvested'].sum(1)
market.head(10)
# create separate dataframe for product and try to visualize
market_county = market.groupby(["State", "County"])["product"].max().sort_values(ascending=False).reset_index()
df3 = pd.DataFrame(market_county)
df3.head()
# visualize by state vs product
plt.subplots(figsize=(20,5))
sns.barplot(
        data=df3,
        x="State",
        y="product",
        palette=['blue', 'red', 'yellow', 'grey'],
        saturation=0.6,
    )

plt.xticks(rotation=90)
# social network user for every state
social_media = usa.groupby(["State", "County"])['Website', 'Facebook', 'Twitter', 'Youtube', 'OtherMedia'].max().reset_index()
social_media = pd.DataFrame(social_media)
social_media.head()
# Website User from different State
website_user = social_media.groupby(["State"])["Website"].sum().sort_values(ascending=False).reset_index()
website_user = pd.DataFrame(website_user)
# Facebook User from different State
facebook_user = social_media.groupby(["State"])["Facebook"].sum().sort_values(ascending=False).reset_index()
facebook_user = pd.DataFrame(facebook_user)
# Twitter User from different State
twitter_user = social_media.groupby(["State"])["Twitter"].sum().sort_values(ascending=False).reset_index()
twitter_user = pd.DataFrame(twitter_user)
# Youtube User from different State
youtube_user = social_media.groupby(["State"])["Youtube"].sum().sort_values(ascending=False).reset_index()
youtube_user = pd.DataFrame(youtube_user)
# OTher Media User from different State
other_user = social_media.groupby(["State"])["OtherMedia"].sum().sort_values(ascending=False).reset_index()
other_user = pd.DataFrame(other_user)
user_web=website_user["State"][website_user["Website"].idxmin()]
print("website user max from {} and from different county {}".format(user_web, website_user["Website"][0]))

user_fb=facebook_user["State"][facebook_user["Facebook"].idxmin()]
print("Facebook user max from {} and from different county {}".format(user_fb, facebook_user["Facebook"][0]))

user_tweet=twitter_user["State"][twitter_user["Twitter"].idxmin()]
print("Twitter user max from {} and from different county {}".format(user_tweet, twitter_user["Twitter"][0]))

user_youtube=youtube_user["State"][youtube_user["Youtube"].idxmin()]
print("Youtube user max from {} and from different county {}".format(user_youtube, youtube_user["Youtube"][0]))

user_other=other_user["State"][other_user["OtherMedia"].idxmin()]
print("OtherMedia user max from {} and from different county {}".format(user_other, other_user["OtherMedia"][0]))
usa.head()
# Total market in different State
total_market = usa.groupby(["State"])["MarketName"].value_counts().groupby(["State"]).sum().sort_values(ascending=False).reset_index()
total_market = pd.DataFrame(total_market)
total_market
data["per capita income"].max()
population = data.groupby(data["State"])["population"].sum().sort_values(ascending=False).reset_index()
population = pd.DataFrame(population)
population