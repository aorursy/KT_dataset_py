import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant reviews.csv')

df = pd.read_csv('../input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')
data.shape, df.shape
data.head()
df.head()
data.isna().sum()
df.isna().sum()
data = data.dropna()
data.isna().sum()
df = df.dropna()
df.isna().sum()
data.shape, df.shape
plt.style.use('fivethirtyeight')
sns.countplot(data=data,x='Rating')
data['Rating'] = data['Rating'].str.replace('Like','5')
sns.countplot(data=data,x='Rating')
data['Rating'] = data['Rating'].astype(float) 
data.head()
data['Time'] = pd.to_datetime(data['Time'])
data['Time'].min(),data['Time'].max()
data.head()
data['Metadata'] =data['Metadata'].str.replace(' Review', ' Reviews')
data.head()
data['reviews'] = data['Metadata'].str.replace('[^0-9,]','').str.split(',').str[0].astype(float)
data['followers'] = data['Metadata'].str.replace('[^0-9,]','').str.split(',').str[1].astype(float)
data.head()
data['reviews'] = data['reviews'].astype(float)
data['followers'].fillna('0', inplace = True)
data['followers'] = data['followers'].astype(float)
data['Time'] = pd.to_datetime(data['Time'])
data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year
data.head()
x = data.groupby(['Restaurant','Rating'])['Rating'].count()
y = x.sort_values(ascending=False).head(11)
y
plt.figure(figsize=(15, 8))
res_rating_5 = data.groupby(['Restaurant','Rating'])['Rating'].count()
top_res_having_5_ratings = res_rating_5.sort_values(ascending = False).head(11)
chart1 = top_res_having_5_ratings[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number_of_5_Ratings')
plt.xlabel('Restaurant_Name')
data['Pictures'].unique()
plt.figure(figsize=(15,8))
res_pic_20 = data.groupby('Restaurant')['Pictures'].max()
sort_pic_values = res_pic_20.sort_values(ascending=False).head(21)
 # end to beginning, counting down by 1
charts = sort_pic_values[::-1].plot.bar()
for p in charts.patches:
    charts.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number_of_Pictures_taken_in_restaurants')
plt.xlabel('Restaurant_Name')
plt.style.use('dark_background')
plt.figure(figsize=(10,7))
top_10_res = data.groupby('Restaurant')['Rating'].mean()
top_10_res_ratings = top_10_res.sort_values(ascending=False).head(10)
chart = top_10_res_ratings[::-1].plot.bar()
for p in chart.patches:
    chart.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Ratings of Restuarants')
plt.xlabel('Restaurant_Name')
data.head()
plt.figure(figsize=(15,8))
top_10_reviewrs = data.groupby('Reviewer')['followers'].sum()
top_10_rev_followers = top_10_reviewrs.sort_values(ascending=False).head(10)
chart1 = top_10_rev_followers[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number of Followers')
plt.xlabel('Reviewer Name')
plt.figure(figsize=(15, 4))
res_avg_rating = data.groupby(['Restaurant', 'Year'])['Rating'].mean()
top10_res = res_avg_rating.sort_values(ascending = False).head(10)
chart2 = top10_res[::-1].plot.bar()
for p in chart2.patches:
    chart2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Rating')
plt.xlabel('Restaurant_Name')
from wordcloud import WordCloud

plt.figure(figsize=(15, 4))
ip_string = ' '.join(data['Review'].dropna().to_list())

wc = WordCloud(background_color='white').generate(ip_string.lower())
plt.imshow(wc)
