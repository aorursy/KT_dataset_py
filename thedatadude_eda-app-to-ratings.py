# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import scipy.stats as stats

# Any results you write to the current directory are saved as output.
#read in and examine 

df_raw = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')

df = df_raw

df.head()
df.info()
df = df.drop(["URL", "Icon URL"], axis=1)
df["Average User Rating"].value_counts()
df = df[~df["Average User Rating"].isna()]
df["Primary Genre"].value_counts()
df = df.drop("Primary Genre", axis=1)
df.head()
plt.figure(figsize=(16,15))

tmp = df[["User Rating Count", "Price", "Size"]]

for j in range(3):

    plt.subplot(3,2,2*j+1)

    sns.distplot(tmp[tmp.columns[j]])

    plt.title(tmp.columns[j] + " Histogram")

    plt.subplot(3,2,2*j+2)

    sns.boxplot(y=tmp[tmp.columns[j]])

    plt.title(tmp.columns[j] +" Boxplot")
df["is_Paid"] = df.Price.map({0:0}).fillna(1)

df["has_Subtitle"] = df.Subtitle.fillna(0).map({0:0}).fillna(1)

df["num_Languages"] = df.Languages.str.count(",")+1

df["description_Length"] = df.Description.str.len()

df["log_Size"] = np.log(df.Size)

df["num_Genres"] = df.Genres.str.count(",")+1

df["has_In-app_Purchases"] = df["In-app Purchases"].fillna(0).map({0:0}).fillna(1)

df["age_of_App"]=(pd.Timestamp.today() - pd.to_datetime(df["Original Release Date"])).dt.days

df["time_Since_Update"]=(pd.Timestamp.today() - pd.to_datetime(df["Current Version Release Date"])).dt.days
df.head()
plt.figure(figsize=(36,28))

plt.subplot(431)

sns.countplot(df["Average User Rating"])

plt.title("Average User Rating Bar")

plt.subplot(432)

sns.countplot(df["is_Paid"])

plt.title("is_Paid Bar")

plt.subplot(433)

sns.countplot(df["has_In-app_Purchases"])

plt.title("has_In-app_Purchases Bar")



plt.subplot(434)

sns.countplot(df["Age Rating"])

plt.title("Age Rating Bar")



plt.subplot(435)

sns.distplot(df["log_Size"])

plt.title("log_size Histogram")



plt.subplot(436)

sns.distplot(df["description_Length"])

plt.title("description_Length Histogram")



plt.subplot(437)

sns.distplot(df["num_Languages"].dropna())

plt.title("num_Languages Histogram")



plt.subplot(438)

sns.countplot(df["has_Subtitle"])

plt.title("has_Subtitle Bar")



plt.subplot(439)

sns.countplot(df["num_Genres"].dropna())

plt.title("num_Genres Histogram")



plt.subplot(4,3,10)

sns.distplot(df["age_of_App"].dropna())

plt.title("afe_of_App Histogram")



plt.subplot(4,3,11)

sns.distplot(df["time_Since_Update"].dropna())

plt.title("time_Since_Update Histogram");
plt.figure(figsize = (24,7))

plt.subplot(121)

tbl = pd.crosstab(df.is_Paid,df["Average User Rating"])

tbl = (tbl.T/tbl.T.sum(axis=0)).T

sns.heatmap(tbl, cmap = 'plasma',square=True)

plt.subplot(122)

for_bar = tbl.reset_index().melt(id_vars=["is_Paid"])

sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["is_Paid"])
plt.figure(figsize = (24,7))

plt.subplot(121)

tbl = pd.crosstab(df["has_In-app_Purchases"],df["Average User Rating"])

tbl = (tbl.T/tbl.T.sum(axis=0)).T

sns.heatmap(data=tbl, cmap = 'plasma',square=True)

plt.subplot(122)

for_bar = tbl.reset_index().melt(id_vars=["has_In-app_Purchases"])

sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["has_In-app_Purchases"])
plt.figure(figsize = (24,7))

plt.subplot(121)

tbl = pd.crosstab(df["Age Rating"],df["Average User Rating"])

tbl = (tbl.T/tbl.T.sum(axis=0)).T

sns.heatmap(data=tbl, cmap = 'plasma',square=True)

plt.subplot(122)

for_bar = tbl.reset_index().melt(id_vars=["Age Rating"])

sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["Age Rating"])
plt.figure(figsize = (24,7))

plt.subplot(121)

tbl = pd.crosstab(df["has_Subtitle"],df["Average User Rating"])

tbl = (tbl.T/tbl.T.sum(axis=0)).T

sns.heatmap(data=tbl, cmap = 'plasma',square=True)

plt.subplot(122)

for_bar = tbl.reset_index().melt(id_vars=["has_Subtitle"])

sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["has_Subtitle"])
plt.figure(figsize = (24,7))

plt.subplot(121)

tbl = pd.crosstab(df["num_Genres"],df["Average User Rating"])

tbl = (tbl.T/tbl.T.sum(axis=0)).T

sns.heatmap(data=tbl, cmap = 'plasma',square=True)

plt.subplot(122)

for_bar = tbl.reset_index().melt(id_vars=["num_Genres"])

sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["num_Genres"])
plt.figure(figsize=(36,14))

plt.subplot(231)

sns.boxplot(y=df.log_Size,x=df["Average User Rating"])

plt.subplot(232)

sns.boxplot(y=df["num_Languages"],x=df["Average User Rating"])

plt.subplot(233)

sns.boxplot(y=df["description_Length"],x=df["Average User Rating"])

plt.subplot(234)

sns.boxplot(y=df["age_of_App"],x=df["Average User Rating"])

plt.subplot(235)

sns.boxplot(y=df["time_Since_Update"],x=df["Average User Rating"])