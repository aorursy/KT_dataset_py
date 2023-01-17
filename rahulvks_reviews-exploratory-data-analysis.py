import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer

import datetime as dt

import re

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

data = pd.read_csv("../input/7282_1.csv",encoding='utf-8')

print('Data Size', data.shape)
data.head(1).T
data.isnull().sum()
#Drop NA Columns

data = data.drop(['reviews.doRecommend','reviews.id'],axis=1)
country = data['country'].value_counts()

sns.barplot(country.index,country.values)
#Hotel Name

hotel_name = data['name'].value_counts()

hotel_name[:20]
plt.rcParams['figure.figsize'] = (8, 5.0)

scores = pd.DataFrame({"Ratings":data["reviews.rating"]})

scores.hist(bins=20)
data['Date'] = pd.to_datetime(data['reviews.dateAdded'], errors='coerce')

data['new_date'] = [d.date() for d in data['Date']]

data['new_time'] = [d.time() for d in data['Date']]

data['day'] = pd.DatetimeIndex(data['new_date']).day 

data['month'] = pd.DatetimeIndex(data['new_date']).month

data['year'] = pd.DatetimeIndex(data['new_date']).year 

data = data.drop(['Date'],axis=1)
Review_Day_Count = data['day'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Review_Day_Count.index, Review_Day_Count.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Average Order By Days")

plt.show()



Reviews_Count_Month = data['month'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Reviews_Count_Month.index, Reviews_Count_Month.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Average Order By Months")

plt.show()



Reviews_Year = data['year'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Reviews_Year.index, Reviews_Year.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Average Order By Year")

plt.show()
User_Ferq=data['reviews.username'].value_counts()[:25]

sns.barplot(User_Ferq.index,User_Ferq.values)

plt.ylabel('User_Name_Count')

plt.xlabel('User_Name')

plt.xticks(rotation='vertical')

plt.show()



User_Ferq.plot()
City_Counts = data['city'].value_counts()[:25]

sns.barplot(City_Counts.index,City_Counts.values)

plt.ylabel('Reviews By City')

plt.xlabel('City Name')

plt.xticks(rotation='vertical')

plt.show()



Province_Counts = data['province'].value_counts()[:25]

sns.barplot(Province_Counts.index,Province_Counts.values)

plt.ylabel('Reviews By Province')

plt.xlabel('Province Code')

plt.xticks(rotation='vertical')

plt.show()