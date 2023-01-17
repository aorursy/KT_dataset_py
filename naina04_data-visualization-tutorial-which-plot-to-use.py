# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Path="../input/google-play-store-apps/"

UserReviews=pd.read_csv(Path+'googleplaystore_user_reviews.csv')

UserReviews.head()
UserReviews.info()
UserReviews.shape
UserReviews.columns
UserReviews.Translated_Review.isnull().value_counts()
gPlay=pd.read_csv(Path+'googleplaystore.csv')

gPlay.head()
gPlay.info()
gPlay.Rating.isnull().value_counts()
#Show statistical information 



gPlay.describe()
UserReviews.describe()
Cnt=gPlay.Category.value_counts()

Cnt
plt.figure(figsize=(16,10))

sns.barplot(x=Cnt[:10].index, y=Cnt[:10].values,data=gPlay)

plt.xlabel("App Categories",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.title("Top 10 App categories",fontsize=15)

plt.xticks(rotation=45)

plt.show()
Cnt1=gPlay.Genres.value_counts()

plt.figure(figsize=(21,10))

sns.barplot(x=Cnt1[:15].index,y=Cnt1[:15].values)

plt.xticks(rotation=45)

plt.xlabel("Genres")

plt.ylabel("Count")

plt.title("Top 15 Genres")

plt.show()
k=gPlay.Category.value_counts().index[:10]

k
Cnt_category=gPlay.Category.value_counts().values[:10]



plt.figure(figsize=(20,10))

sns.barplot(y=gPlay.Installs.value_counts().index[:10],x=Cnt_category,hue=k,data=gPlay)

plt.yticks(rotation=0)

plt.show()
UserReviews.Sentiment.value_counts(dropna=True)



plt.figure(figsize=(10,5))

sns.countplot(x="Sentiment",data=UserReviews)

plt.xticks(rotation=45)

plt.xlabel("Sentiments",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.title("User Sentiment Chart",fontsize=18)

plt.show()
UserReviews.Sentiment_Polarity.value_counts()

k=UserReviews.Sentiment_Polarity.mean()

k
#Fill the missing values in Sentiment Polarity with its mean which is 0.18214631382968358

UserReviews['Sentiment_Polarity']=UserReviews.Sentiment_Polarity.fillna(k)
#Check for Null Values

UserReviews.Sentiment_Polarity.isnull().sum()
plt.figure(figsize=(16,10))

plt.hist(UserReviews["Sentiment_Polarity"],bins=10,edgecolor="black")

plt.show()
gPlay.Rating.describe()
plt.figure(figsize=(30,10))

sns.boxplot(y="Rating",x="Category",hue="Type",data=gPlay , palette="Set2", )

plt.xticks(rotation=90)

plt.xlabel("App Categories", fontsize=12)

plt.ylabel("Rating",fontsize=12)

plt.title("Rating by App category",fontsize=15)

plt.show()

gPlay1=gPlay.rename(columns={"Content Rating":"Content_Rating"})

gPlay1.Content_Rating.unique()
plt.figure(figsize=(30,10))

sns.boxplot(y="Rating",x="Content_Rating",hue="Type",data=gPlay1, palette="Set3", )

plt.xticks(rotation=45)

plt.xlabel("Content Rating categories", fontsize=12)

plt.ylabel("Rating",fontsize=12)

plt.title("Rating by Content Rating",fontsize=15)

plt.show()

plt.figure(figsize=(10,18))

sns.violinplot(x="Rating",y="Category", data=gPlay, palette="Set3")

plt.xlabel("Ratings",fontsize=15)

plt.ylabel("App categories",fontsize=15)

plt.title("Ratings by App category",fontsize=15)

plt.show()
gPlay.columns
# Point Plot

cat_count=gPlay.Category.value_counts().index[:10]

rating_count=gPlay.Rating.value_counts().index[:10]



cat_count