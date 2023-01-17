# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
data = pd.read_csv("../input/Tweets.csv")
data.head(1)
data.airline.value_counts()
data.columns
pd.Series(data["airline"]).value_counts().plot(kind = "bar",figsize=(8,6),fontsize=12,rot = 0, title = "Airline")
data.airline_sentiment.value_counts()

pd.Series(data["airline_sentiment"]).value_counts().plot(kind = "bar",figsize=(8,6),fontsize=12,rot = 0, title = "airline_sentiment")
pd.Series(data["airline"]).value_counts().plot(kind = "pie",figsize=(8,6),fontsize=12,rot = 0, title = "Airline")
pd.Series(data["airline_sentiment"]).value_counts().plot(kind = "pie",figsize=(8,6),fontsize=12,rot = 0, title = "airline_sentiment")
data["tweet_location"].unique
pd.Series(data["negativereason"]).value_counts().plot(kind = "barh",figsize=(10,6),fontsize=12,rot = 0, title = "Negative Reason")
pd.Series(data["negativereason"]).value_counts().head(5).plot(kind = "pie",labels=["Customer Service Issue", "Late Flight", "Can't Tell","Cancelled Flight","Lost Luggage"],
                                                      figsize=(6,6),fontsize=12,rot = 0, title = "Negative Reason")
data.tweet_location.value_counts()  

pd.Series(data["user_timezone"]).value_counts().head(10).plot(kind = "barh",figsize=(8,6),title = "User_Timezone")



pd.Series(data["user_timezone"]).value_counts().head(5).plot(kind = "pie",figsize=(8,6),title = "User_Timezone")

air_sen=pd.crosstab(data.airline, data.airline_sentiment)
air_sen
percentage=air_sen.apply(lambda a: a / a.sum() * 100, axis=1)
percentage
pd.Series(percentage["negative"]).plot(kind = "bar",figsize=(8,6),title = "Most Negative Airlines")

pd.Series(percentage["positive"]).plot(kind = "bar",figsize=(8,6),title = "Most Positive Airlines")

