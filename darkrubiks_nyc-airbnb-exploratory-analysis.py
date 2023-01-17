import pandas as pd 

import seaborn as sns 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import geopandas



data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.isna().sum()
host_names = data['host_name'].dropna()

names = data['name'].dropna()

neighbourhood = data['neighbourhood']
def make_wordcloud(words):



    text = ""

    for word in words:

        text = text + " " + word



    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords,colormap="plasma",width=1920, height=1080,max_font_size=200, max_words=200, background_color="white").generate(text)

    plt.figure(figsize=(20,20))

    plt.imshow(wordcloud, interpolation="gaussian")

    plt.axis("off")

    plt.show()
make_wordcloud(host_names)
make_wordcloud(names)
make_wordcloud(neighbourhood)
crs = {'init':'epsg:4326'}

geometry = geopandas.points_from_xy(data.longitude, data.latitude)

geo_data = geopandas.GeoDataFrame(data,crs=crs,geometry=geometry)
nyc = geopandas.read_file(geopandas.datasets.get_path('nybb'))

nyc = nyc.to_crs(epsg=4326)
fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='id',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Number of Airbnb Listings")

plt.axis('off')
plt.figure(figsize=(10,10))

ax = sns.countplot(data["neighbourhood_group"], palette="plasma" ) 
fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='room_type',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Locations of Room Types")

plt.axis('off')
plt.figure(figsize=(10,10))

ax = sns.countplot(data['room_type'],hue=data['neighbourhood_group'], palette='plasma')
fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='availability_365',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Number of days when listing is available for booking")

plt.axis('off')
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=data, x='neighbourhood_group',y='availability_365',palette='plasma')
plt.figure(figsize=(10,10))

ax = sns.heatmap(data.corr(),annot=True)
data['price'].groupby(data["neighbourhood_group"]).describe().round(2)
data['price'].groupby(data["room_type"]).describe().round(2)
df = data.copy()

df.drop(['id','host_id','name','host_name','last_review','geometry','neighbourhood'],axis=1,inplace=True)

df.dropna(inplace=True)

df_dummies = pd.get_dummies(df)
y = df_dummies['price']

df_dummies.drop(['price'],axis=1,inplace=True)

X = df_dummies
from sklearn import linear_model



reg = linear_model.LinearRegression().fit(X, y)

print("R²: {}".format(reg.score(X, y)))