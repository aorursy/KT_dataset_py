# Imports and Basic Data Cleaning (Drop Duplicates and NaN data)
import numpy as np
import pandas as pd 
import os
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

df = pd.read_csv('/kaggle/input/windows-store/msft.csv')
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

print(df.head())
print(df.info())
from numpy import mean

plt.figure(figsize=(10,5))
sns.distplot(df.Rating, kde=False)
plt.ylabel('Count')
plt.title('Distribution of Overall Rating')
plt.show()

plt.figure(figsize=(10,5))
sns.distplot(df["No of people Rated"], kde=True)
plt.ylabel('Frequency')
plt.xlabel('Number of Ratings per App')
plt.title('Distribution of Number of Ratings per App')
plt.show()

plt.figure(figsize=(10,5))
tot = df.shape[0]
vc = df['Price'].value_counts()
num_free = vc['Free']
num_cost = tot - num_free
slices = [num_free, num_cost]
labeling = ['Free', 'Not Free']
explode = [0.1, 0.2]
plt.pie(slices,explode=explode,shadow=True,autopct='%1.1f%%',labels=labeling,wedgeprops={'edgecolor':'black'})
plt.title('Free vs. Not Free Apps')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
g = sns.countplot(x='Category',data=df)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Distribution of Category Types')
plt.ylabel('Number of Apps')
plt.show()

sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')
g = sns.catplot(x="Rating", y="Category", kind="box", showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, estimator='mean', order=df.Category.value_counts().iloc[:25].index)
g.set(xlim=(0, 5.5))
g.fig.set_size_inches(20, 10)
g.ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5], minor=True)
plt.title("Ratings by Category - Ordered by most common Categories")
plt.show()

df_cat = df.groupby("Category").mean().sort_values(by = 'Rating', ascending=False)
g = sns.catplot(x="Rating", y="Category", kind="box", showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_cat.iloc[:25].index)
g.set(xlim=(0, 5.5))
g.fig.set_size_inches(20, 10)
g.ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5], minor=True)
plt.title("Ratings by Category - Oredered by highest mean Rating")
plt.show()

print("Mean Ratings by Category: ")
print(df_cat.Rating.to_string())

from wordcloud import WordCloud,STOPWORDS 

plt.style.use('fivethirtyeight')
stopwords = set(STOPWORDS) 
stop_word= list(stopwords) + ['http','co','https','wa','amp','û','Û','HTTP','HTTPS']

fig, (ax1) = plt.subplots(1, 1, figsize=[26, 8])
wordcloud1 = WordCloud( background_color='white',stopwords = stop_word,
                        width=600,
                        height=400).generate(" ".join(df['Name']))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('App Title WordCloud',fontsize=40)
plt.show()
def month_lookup(x):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    return months[x-1]

def day_lookup(x):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[x]

def year_lookup(x):
    return str(x) + 'x'

df['day'] = pd.DatetimeIndex(df['Date'], dayfirst=True).day
df['month'] = pd.DatetimeIndex(df['Date'], dayfirst=True).month
df['month_str'] = df['month'].apply(lambda x: month_lookup(x))
df['year'] = pd.DatetimeIndex(df['Date'], dayfirst=True).year
df['year_str'] = df['year'].apply(lambda x: year_lookup(x))
df['day_of_week'] = pd.DatetimeIndex(df['Date'], dayfirst=True).dayofweek 
df['day_of_week_str'] = df['day_of_week'].apply(lambda x: day_lookup(x))
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10,5))
g = sns.countplot(x='day_of_week_str',data=df, order=days)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Distribution of Publish Date by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Apps')
plt.show()

df_day = df.groupby("day_of_week_str").mean().sort_values(by = 'Rating', ascending=False)
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')
g = sns.catplot(x="Rating", y="day_of_week_str", kind="box", showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order=df_day.iloc[:25].index)
g.set(xlim=(0, 5.5))
g.fig.set_size_inches(10, 10)
g.ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5], minor=True)
plt.title("Ratings by Week - Ordered by Mean Rating")
plt.ylabel("Day of the Week")
plt.show()

print("Mean Ratings by Day of Week: ")
print(df_day.Rating.to_string())


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.figure(figsize=(10,5))
g = sns.countplot(x='month_str',data=df, order=months)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Distribution of Publish Date by Month')
plt.xlabel('Month')
plt.ylabel('Number of Apps')
plt.show()

df_month = df.groupby("month_str").mean().sort_values(by = 'Rating', ascending=False)
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')
g = sns.catplot(x="Rating", y="month_str", kind="box", showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"},data=df, order=df_month.iloc[:25].index)
g.set(xlim=(0, 5.5))
g.fig.set_size_inches(10, 10)
g.ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5], minor=True)
plt.title("Ratings by Month - Oredered by Mean Rating")
plt.ylabel("Month")
plt.show()

print("Mean Ratings by Month: ")
print(df_month.Rating.to_string())
years = ['2010x', '2011x', '2012x', '2013x', '2014x', '2015x', '2016x', '2017x', '2018x', '2019x']

plt.figure(figsize=(10,5))
g = sns.countplot(x='year_str',data=df, order=years)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Distribution of Publish Date by Year')
plt.xlabel('Year')
plt.ylabel('Number of Apps')
plt.show()


df_year = df.groupby("year_str").mean().sort_values(by = 'Rating', ascending=False)
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')
g = sns.catplot(x="Rating", y="year_str",kind="box",showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, data=df, order = df_year.iloc[:25].index)
g.set(xlim=(0, 5.5))
g.fig.set_size_inches(10, 10)
g.ax.set_xticks([1,1.5,2,2.5,3,3.5,4,4.5,5], minor=True)
plt.title("Ratings by Year - Ordered by Mean Rating")
plt.ylabel("Year")
plt.show()

print("Mean Ratings by Year: ")
print(df_year.Rating.to_string())

