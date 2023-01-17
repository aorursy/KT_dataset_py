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
# import necessary libraries for visualization.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# import sklearn text_processing library
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

sns.set()

%matplotlib inline
df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
df.head(5)
# DataFrame Column can be found by using .dtypes method of DataFrame object
df.dtypes
# In these case, we can use to_datetime method of pandas
df["date_added"] = pd.to_datetime(df["date_added"])
df.dtypes
df.info()
# show_id should be unique,so let's check its uniqueness 
# nunique method can show the number of unique values in dataframe
df.nunique()
# First of all, let's check the value of type column
df["type"].head()
sns.countplot("type", data=df)
plt.title("Show Type Count in Netflix dataset")
df["title"]
# CountVectorizer?
# Use Bag of Words, and vectorize all the words.
countvectorizer = CountVectorizer(stop_words="english")
bow = countvectorizer.fit_transform(df["title"])
bow.toarray(), bow.shape
# Get feature names
feature_names = countvectorizer.get_feature_names()

# View some feature names
feature_names[150:160]
# Create data frame (column: words in title, row: each row of original dataframe)
bow_result_df = pd.DataFrame(bow.toarray(), columns=feature_names)
bow_result_df.head()
# Let's see the word that is used for 20 times.
frequent_word_df = pd.DataFrame(bow_result_df.sum(), bow_result_df.columns)
frequent_word_df = frequent_word_df.rename(columns={0:"count"})
frequent_word_df = frequent_word_df[frequent_word_df["count"] > 20]
frequent_word_df.head(5)
frequent_word_sorted_df = frequent_word_df.sort_values("count", ascending=False)
frequent_word_sorted_df.head()
plt.figure(figsize=(12, 4))
sns.barplot(frequent_word_sorted_df.index, frequent_word_sorted_df["count"])
plt.xticks(rotation=60)
plt.xlabel("Word")
plt.title("Word Count of Movie Titles")
# How many NaN is included here?
df["director"].isnull().sum()
# pick up directors who directs more than twice.
director_df = df["director"]
director_removed_nan_df = director_df.dropna()
director_removed_nan_df.head()
# I want a dictionary which contains how many times does each director appear?
# Key: Director(s) Name, Value: Appearance Count of each directors
director_count = {}

for i in director_removed_nan_df.index:
    director_count.setdefault(director_removed_nan_df[i], 0)
    director_count[director_removed_nan_df[i]] += 1
# In the director_count dictionary, we pick up the frequent directors.
# Criteria: Appearance Count is 6 times and above.
frequent_director_count = {}

for key,value in director_count.items():
    if value >= 6:
        frequent_director_count.setdefault(key, value)
frequent_director_count
sorted_dict = sorted(frequent_director_count.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])
plt.figure(figsize=(12,4))
sns.barplot(x, y)
plt.xticks(rotation=90)
plt.yticks(np.arange(0, 20, step=1))
plt.title("Director Count in Netflix")
plt.ylabel("Count")
plt.xlabel("Director(s)")
df["country"][0]
df["country"].dropna()[0].split(",")
# Let's create the dictionary that contains how many times does each country appear?
# Key: Country Name, Value: Times that each country appears.
frequent_country = {}

for i in df["country"].dropna().index:
    country_list = df["country"].dropna()[i].split(",")
    for country in country_list:
        frequent_country.setdefault(country, 0)
        frequent_country[country] += 1
# Sort it by using sorted function in dictionary.
sorted_dict = sorted(frequent_country.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])
# Plot the result of the countries which is in top 20.
plt.figure(figsize=(12, 4))
sns.barplot(x[:20], y[:20])
plt.xticks(rotation=90)
plt.xlabel("Country")
plt.ylabel("Count")
plt.title("Country Count in Netflix")
df["date_added"]
df["date_added"].isnull().sum()
date_count_series = df.groupby("date_added")["show_id"].count()
date_count_series.head()
plt.figure(figsize=(16, 4))
plt.plot(date_count_series.index, date_count_series.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.figure(figsize=(16, 4))
plt.plot(date_count_series[date_count_series.index >= "2016-01-01"].index, date_count_series[date_count_series.index >= "2016-01-01"].values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.figure(figsize=(16, 4))
plt.plot(date_count_series[date_count_series.index >= "2019-01-01"].index, date_count_series[date_count_series.index >= "2019-01-01"].values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
date_type_df = pd.DataFrame(date_count_series)
date_type_df["day_type"] = date_count_series.index.dayofweek
date_type_df.head(5)
grouped_date_type_series = date_type_df.groupby("day_type").count()
grouped_date_type_series
day_type=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.bar(x=grouped_date_type_series.index, height=grouped_date_type_series["show_id"])
plt.xticks(np.arange(7), labels=day_type)
plt.ylabel("Count")
plt.title("Day type Count in Netflix")
release_year_series = df.groupby("release_year")["show_id"].count()
release_year_series.index = pd.to_datetime(release_year_series.index, format="%Y")
release_year_series.head(5)
plt.figure(figsize=(16, 4))
plt.plot(release_year_series.index, release_year_series.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
plt.xticks(rotation=60)
plt.title("Publish Year in all shows")
plt.ylabel("Count")
plt.xlabel("Year")
df["rating"].nunique()
sns.countplot(df["rating"])
plt.xticks(rotation=90)
plt.title("Rating Count in Netflix")
df["duration"]
movie_duration_series = pd.DataFrame(df[df["type"] == "Movie"]["duration"])
movie_duration_series.head(5)
movie_duration_series = movie_duration_series.replace("(\d*) min", r"\1", regex=True)
movie_duration_series["duration"] = movie_duration_series["duration"].astype("int64")
movie_duration_series.head()
plt.hist(movie_duration_series["duration"], bins=20)
plt.xlabel("duration")
plt.ylabel("count")
plt.title("Movies duration histgram in Netflix")
plt.hist(movie_duration_series["duration"], bins=20, density=True)
plt.xlabel("duration")
plt.ylabel("count")
plt.title("Relative Frequency Distribution of Movies duration in Netflix")
movie_duration_series.describe()
df["listed_in"]
# Key: show classification, Value: Count 
frequent_listed_in = {}

for i in df["listed_in"].index:
    listed_in_list = df["listed_in"][i].split(",")
    for listed_in in listed_in_list:
        frequent_listed_in.setdefault(listed_in, 0)
        frequent_listed_in[listed_in] += 1

sorted_dict = sorted(frequent_listed_in.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])
plt.figure(figsize=(12, 4))
sns.barplot(x[:20], y[:20])
plt.xticks(rotation=90)
plt.xlabel("Show Type")
plt.ylabel("Count")
plt.title("Show Type Count in Netflix")
df["description"][0]
# Use Bag of Words, and vectorize all the words.
countvectorizer = CountVectorizer(stop_words="english")
bow = countvectorizer.fit_transform(df["description"])
bow.toarray(), bow.shape
# Get feature names
feature_names = countvectorizer.get_feature_names()

# View feature names
feature_names[1500:1510]
# Create data frame (column: words in description, row: each row of original dataframe)
bow_result_df = pd.DataFrame(bow.toarray(), columns=feature_names)
bow_result_df.head()
# Let's see the word that is used for 200 times.
frequent_word_df = pd.DataFrame(bow_result_df.sum(), bow_result_df.columns)
frequent_word_df = frequent_word_df.rename(columns={0:"count"})
frequent_word_df = frequent_word_df[frequent_word_df["count"] > 200]
frequent_word_df.head(5)
frequent_word_sorted_df = frequent_word_df.sort_values("count", ascending=False)
frequent_word_sorted_df.head()
plt.figure(figsize=(12, 4))
sns.barplot(frequent_word_sorted_df.index, frequent_word_sorted_df["count"])
plt.xticks(rotation=60)
plt.xlabel("Word")
plt.title("Word Count of Movie Description")
wordcloud = WordCloud(background_color="white")
wordcloud.generate(" ".join(df["description"]))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Show Description WordCloud in Netflix dataset")