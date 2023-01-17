import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

params={"figure.facecolor":(0,0,0,0),

        "axes.facecolor":(1,1,1,1),

        "savefig.facecolor":(0,0,0,0,),

        "axes.titlesize":14}

plt.rcParams.update(params)
df=pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")

df.head(2)
df.drop(["Unnamed: 0","id","currency","rating_count_ver","user_rating_ver","ver","sup_devices.num","ipadSc_urls.num","lang.num","vpp_lic"],axis=1,inplace=True)
df.head()
df.info()
sns.countplot(df["user_rating"],palette="Purples")

plt.xlabel("User Ratings")

plt.ylabel("Count")

plt.title("Frequency of Ratings")
sns.countplot(df["user_rating"].loc[df["rating_count_tot"]>0],palette="Purples")

plt.xlabel("User Ratings")

plt.ylabel("Count")

plt.title("Frequency of Ratings")
df.sort_values(by=["user_rating","rating_count_tot"],ascending=False).head(10)
df.loc[df["rating_count_tot"]>0].sort_values(by=["user_rating","rating_count_tot"]).head(10)
plt.figure(figsize=(12,4))

sns.distplot(df["price"],bins=100,rug=True,color="deepskyblue")

plt.xlabel("Price (USD)")

plt.ylabel("Frequency")

plt.title("App Price Distribution")
df.loc[df["price"]>50].sort_values(by="price",ascending=False)
genres=df.groupby(["prime_genre"]).agg({"prime_genre":"size","user_rating":"mean"})



fig,ax1=plt.subplots(figsize=(10,4))

sns.barplot(x=genres.index,y=genres["prime_genre"],color="green",ax=ax1)

ax1.set_ylabel("Count",color="green")

for label in ax1.get_yticklabels():

    label.set_color("green")

ax1.set_xticklabels(genres.index,rotation=90)

ax1.set_xlabel("")

    

ax2=ax1.twinx()

sns.scatterplot(x=genres.index,y=genres["user_rating"],color="orchid",ax=ax2)

ax2.set_ylabel("Average Rating",color="orchid")

for label in ax2.get_yticklabels():

    label.set_color("orchid")

ax2.set_xticklabels(genres.index,rotation=90)

ax2.set_xlabel("")



plt.suptitle("Genre Count and Rating",fontsize=14,y=0.94)
sns.countplot(df["cont_rating"],order=df["cont_rating"].value_counts().index,palette="OrRd")

plt.xlabel("Content Rating")

plt.ylabel("Count")

plt.title("Frequency of Content Rating")
df.loc[df["cont_rating"]=="17+"].sort_values(by=["user_rating","rating_count_tot"],ascending=False).head()
plt.figure(figsize=(12,4))

sns.boxplot(df["size_bytes"],color="#FFFF66")

plt.xlabel("Size (bytes)")

plt.title("Distribution of App Size")
df.loc[df["size_bytes"]>3500000000].sort_values(by=["user_rating","rating_count_tot"],ascending=False).head()