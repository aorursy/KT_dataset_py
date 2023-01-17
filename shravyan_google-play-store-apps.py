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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import datetime
gps = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

review = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
gps.head()
gps.info()
gps[gps.isnull().any(axis=1)]
#pd.to_numeric(gps["Reviews"])
gps.loc[10472,"Reviews"]
gps.loc[10472,"Reviews"]='3000000'
gps.iloc[10472]["Reviews"]
gps["Reviews"]=pd.to_numeric(gps["Reviews"])
gps["Size"].unique()
gps["Size"].value_counts()
gps["to_del"]=gps["Size"].apply(lambda x:x[-1])
gps["Size"]=gps["Size"].apply(lambda x:x[:-1])
gps["Size"]=pd.to_numeric(gps["Size"],errors='coerce')
gps.loc[gps["to_del"]=="M","Size"]=gps[gps["to_del"]=="M"]["Size"].apply(lambda x: x*1000)
gps["Size"]
gps.rename(columns={"Size":"Size in kb"})
gps.drop("to_del",axis=1,inplace=True)
gps["Price"].unique()
gps["Price"]=gps["Price"].apply(lambda x:x[1:])
gps["Price"]=pd.to_numeric(gps["Price"],errors='coerce')
gps["Last Updated"]=pd.to_datetime(gps["Last Updated"],errors="coerce")


gps.hist(figsize=(10,5));
fig=plt.figure(figsize=(20,7))

sns.countplot(gps["Category"]);

plt.xticks(rotation=90)

fig.show()
gps[gps["Size"]==gps["Size"].max()][["App","Size"]]
gps[gps["Last Updated"]==gps["Last Updated"].min()]
gps["Last Updated"].max()
gps[gps["Last Updated"].dt.year.eq(2010)]
fig=plt.figure(figsize=(20,7))

sns.countplot(gps.Installs);



plt.xticks(rotation=90)

fig.show()
gps[gps["Installs"]=="1,000,000,000+"]["App"]
gps["Year"]=gps["Last Updated"].dt.year
gps["Installs1"]=gps["Installs"].apply(lambda x: x[:-1])
gps["Installs1"]=gps["Installs1"].apply(lambda x:x.replace(",",""))
gps["Installs1"]=pd.to_numeric(gps["Installs1"],errors="coerce")
gps.groupby("Year")["Installs1"].max()
#gps[gps["Year"]==2016.0].sort_values("Installs1",ascending=False)

gps2016= gps[gps["Year"]==2016.0]

gps2016[gps2016["Installs1"]==gps2016["Installs1"].max()]["App"]
gps2017= gps[gps["Year"]==2017.0]

gps2017[gps2017["Installs1"]==gps2017["Installs1"].max()]["App"]
gps2018= gps[gps["Year"]==2018.0]

gps2018[gps2018["Installs1"]==gps2018["Installs1"].max()]["App"]
gps[gps["Reviews"]==gps["Reviews"].max()]
sns.countplot(gps["Type"])
gps["Price"].hist(bins=10)
gps[gps["Price"]>350]
gps.info()
sns.pairplot(gps)
sns.heatmap(gps.corr(),annot=True)
gps.head()
gps.groupby("Category")["Rating","Reviews","Size","Installs1"].mean()

#for key,item in Cat:

#    print(Cat.get_group(key), "\n\n")
fig=plt.figure(figsize=(20,7))

sns.barplot(gps["Category"],gps["Reviews"]).set_title("Reviews by Category");

plt.xticks(rotation=90)

fig.show()
fig=plt.figure(figsize=(20,7))

sns.barplot(gps["Category"],gps["Size"]).set_title("App Size by Category");

plt.xticks(rotation=90)

fig.show()
gps.groupby("Content Rating")["Rating","Reviews","Size","Installs1"].mean()
fig=plt.figure(figsize=(10,5))

sns.barplot(gps["Content Rating"],gps["Reviews"]).set_title("Reviews by Content Category");
fig=plt.figure(figsize=(10,5))

sns.barplot(gps["Content Rating"],gps["Installs1"]).set_title("App downloads by Content Category");
gps.groupby("Year")["Rating","Reviews","Size","Installs1"].mean()
fig=plt.figure(figsize=(10,5))

sns.lineplot(gps["Year"],gps["Reviews"]).set_title("Reviews by Year(Last Updated)");
fig=plt.figure(figsize=(10,5))

sns.lineplot(gps["Year"],gps["Size"]).set_title("App Size over the years(Last Updated)");
fig=plt.figure(figsize=(10,5))

sns.lineplot(gps["Year"],gps["Installs1"]).set_title("App downloads over years(Last Updated)");
Version =gps.groupby("Current Ver")["Rating","Reviews","Size","Installs","Year"].mean()
Version[Version["Rating"]<4.0]#
gps.groupby("Android Ver")["Rating","Reviews","Size","Installs"].mean()
gps.describe()
genre= gps.groupby(["Category","Genres"])["Rating","Reviews","Size","Installs1"].mean()
genre[genre["Rating"]<4.0]
gps[gps["Rating"]==gps["Rating"].min()]
gps[gps["Rating"].between(1.1,2.0,inclusive=True)]
genre[genre["Rating"]>4.5]
genre[genre["Reviews"]<40]
genre[genre["Reviews"]>150000]
genre[genre["Size"]<4500]
genre[genre["Size"]>50000]
genre[genre["Installs1"]<3000]
genre[genre["Installs1"]>10000000]
review.head()
review.info()
review.describe()
review.dropna(inplace=True)
review.columns
review1=review.groupby("App")['Sentiment_Polarity',

       'Sentiment_Subjectivity'].mean()
sns.heatmap(review.corr(),annot=True)
review1[review1["Sentiment_Polarity"]>0.5]
review1[review1["Sentiment_Subjectivity"]<0.5]