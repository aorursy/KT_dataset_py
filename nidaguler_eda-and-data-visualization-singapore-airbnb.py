import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import os





import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/singapore-airbnb/listings.csv")
df.head()
df = df.sort_values(by=["price"], ascending=False)

df['rank']=tuple(zip(df.price))

df['rank']=df.groupby('price',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df.head()
df.drop(["rank"],axis=1,inplace=True)
df.reset_index(inplace=True,drop=True)

df.head()
df.info()
df.shape
df.tail()
df.drop(['id','host_id','host_name','last_review'],axis=1,inplace=True)
df.isnull().sum()
def impute_median(series):

    return series.fillna(series.median())
df.reviews_per_month=df["reviews_per_month"].transform(impute_median)
df.isnull().sum()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)

plt.show()
sns.pairplot(df)

plt.show()
fig = plt.figure(figsize = (15,10))

ax = fig.gca()

df.hist(ax=ax)

plt.show()
df.columns
df.nunique()
#room_type - price

result = df.groupby(["room_type"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x='room_type', y="price", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#neighbourhood_group - price

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x='neighbourhood_group', y="price", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
labels = df.neighbourhood_group.value_counts().index

colors = ['green','yellow','orange','pink','red']

explode = [0,0,0,0,0]

sizes = df.neighbourhood_group.value_counts().values



plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Airbnb According to Neighbourhood Group',color = 'blue',fontsize = 15)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x = "neighbourhood_group", y = "price", hue = "room_type", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,18))

sns.lmplot(x='minimum_nights',y='calculated_host_listings_count',hue="neighbourhood_group",data=df)

plt.xlabel('calculated_host_listings_count')

plt.ylabel('minimum_nights')

plt.title('calculated_host_listings_count vs minimum_nights')

plt.show()
df.price.max()
plt.figure(figsize=(15,6))

ax = sns.violinplot(x="neighbourhood_group", y="price",

                    data=df[df.price < 1000],

                    scale="width", palette="Set3")
#neighbourhood_group - reviews_per_month

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x='neighbourhood_group', y="reviews_per_month", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#neighbourhood_group - minimum_nights

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')

sns.barplot(x='neighbourhood_group', y="minimum_nights", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#neighbourhood_group - number_of_reviews

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')

sns.barplot(x='neighbourhood_group', y="number_of_reviews", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()


ax = sns.violinplot(x="room_type", y="price",

                    data=df[df.price > 500],

                    scale="width", palette="Set3")
sns.kdeplot(df['price'])

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('Price Kde Plot')

plt.show()
#neighbourhood_group - reviews_per_month

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x='neighbourhood_group', y="reviews_per_month", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='reviews_per_month',y='price',data=df)

plt.show()
#neighbourhood_group - calculated_host_listings_count

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x='neighbourhood_group', y="calculated_host_listings_count", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='calculated_host_listings_count',y='price',data=df)

plt.show()
#neighbourhood_group - availability_365

plt.figure(figsize=(15,6))

result = df.groupby(["neighbourhood_group"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x='neighbourhood_group', y="availability_365", data=df, order=result['neighbourhood_group']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='availability_365',y='price',data=df)

plt.show()
df.price.describe().T
labels = df.room_type.value_counts().index

colors = ['orange','yellow','red']

explode = [0,0,0]

sizes = df.room_type.value_counts().values



plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Airbnb According to Room Type',color = 'blue',fontsize = 15)

plt.show()
#room_type - minimum_nights

result = df.groupby(["room_type"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')

sns.barplot(x='room_type', y="minimum_nights", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='minimum_nights',y='price',data=df)

plt.show()
#room_type - number_of_reviews

result = df.groupby(["room_type"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')

sns.barplot(x='room_type', y="number_of_reviews", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
sns.lineplot(x='number_of_reviews',y='price',data=df)

plt.show()
#room_type - reviews_per_month

result = df.groupby(["room_type"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x='room_type', y="reviews_per_month", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#room_type - calculated_host_listings_count

result = df.groupby(["room_type"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x='room_type', y="calculated_host_listings_count", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#room_type - availability_365

result = df.groupby(["room_type"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x='room_type', y="availability_365", data=df, order=result['room_type']) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.show()
#neighbourhood - price

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x=df.price[:50], y=df.neighbourhood[:50]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
#neighbourhood - minimum_nights

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['minimum_nights'].aggregate(np.median).reset_index().sort_values('minimum_nights')

sns.barplot(x=df.minimum_nights[:500], y=df.neighbourhood[:500]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
#neighbourhood - number_of_reviews

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['number_of_reviews'].aggregate(np.median).reset_index().sort_values('number_of_reviews')

sns.barplot(x=df.number_of_reviews[:500], y=df.neighbourhood[:500]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
#neighbourhood - reviews_per_month

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['reviews_per_month'].aggregate(np.median).reset_index().sort_values('reviews_per_month')

sns.barplot(x=df.reviews_per_month[:50], y=df.neighbourhood[:50]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
#neighbourhood - calculated_host_listings_count

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['calculated_host_listings_count'].aggregate(np.median).reset_index().sort_values('calculated_host_listings_count')

sns.barplot(x=df.calculated_host_listings_count[:100], y=df.neighbourhood[:100]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
#neighbourhood - availability_365

plt.figure(figsize=(18,8))

#result = df.groupby(["neighbourhood"])['availability_365'].aggregate(np.median).reset_index().sort_values('availability_365')

sns.barplot(x=df.availability_365[:100], y=df.neighbourhood[:100]) #formerly: sns.barplot(x='Id', y="Speed", data=df, palette=colors, order=result['Id'])

plt.xticks(rotation=90)

plt.show()
df_rich_hosts=pd.DataFrame(df.iloc[:,0:2])

df_rich_hosts['latitude']=df.iloc[:,3]

df_rich_hosts['longitude']=df.iloc[:,4]

df_rich_hosts['room_type']=df.iloc[:,5]

df_rich_hosts['price']=df.iloc[:,6]

df_rich_hosts.head()
df.room_type.unique()
df[df.room_type=="Private room"].describe().T
plt.figure(figsize=(15,6))

sns.scatterplot(df_rich_hosts.longitude,df_rich_hosts.latitude,hue=df_rich_hosts.neighbourhood_group)

plt.ioff()
plt.figure(figsize=(15,6))

sns.scatterplot(df_rich_hosts.longitude,df_rich_hosts.latitude,hue=df_rich_hosts.room_type)

plt.ioff()
print(df.latitude.max())

print(df.latitude.min())

print(df.longitude.max())

print(df.longitude.min())
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df_rich_hosts.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="yellow").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
import folium

from folium.plugins import HeatMap

m=folium.Map([1.24387,103.973],zoom_start=11)

HeatMap(df_rich_hosts[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)