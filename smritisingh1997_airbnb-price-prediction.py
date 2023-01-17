import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from scipy.special import boxcox1p



import folium

import folium.plugins

from folium.plugins import HeatMap

from folium.plugins import MarkerCluster

from folium import plugins



import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_rows = None

pd.options.display.max_columns = None
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.shape
data.dtypes
data.duplicated().sum()
total = data.isnull().sum().sort_values(ascending = False)

percent = (((data.isnull().sum()) * 100) / data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys=['Total', 'Percent'], sort=False).sort_values('Total', ascending=False)
missing_data.head()
data.isnull().sum()
data.describe().T
data.corr().style.background_gradient(cmap='coolwarm')
corr = data.corr() 

plt.figure(figsize=(15, 10))

sns.heatmap(corr, annot=True)
data.columns
data.head()
data.availability_365.dtypes
data['status_availability_365'] = 'Worst'

data.loc[data['availability_365'] >200, 'status_availability_365'] = "Best"

data.loc[(data['availability_365'] > 100) & (data['availability_365'] < 200), 'status_availability_365'] = 'Good'

data.loc[(data['availability_365'] > 0) & (data['availability_365'] < 100), 'status_availability_365'] = 'Okay'
data.head()
data.status_availability_365.value_counts()
data['average_price'] = data['minimum_nights'] / data['price']
from scipy.stats import norm

from scipy import stats
sns.distplot(data['availability_365'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['availability_365'], plot = plt)
sns.distplot(data['calculated_host_listings_count'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['calculated_host_listings_count'], plot = plt)
data['reviews_per_month'] = data['reviews_per_month'].replace(np.nan, 0)
sns.distplot(data['reviews_per_month'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['reviews_per_month'], plot = plt)
sns.distplot(data['number_of_reviews'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['number_of_reviews'], plot = plt)
sns.distplot(data['latitude'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['latitude'], plot = plt)
sns.distplot(data['longitude'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['longitude'], plot = plt)
sns.distplot(data['minimum_nights'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['minimum_nights'], plot = plt)
data.average_price.replace([np.inf, -np.inf], np.nan, inplace = True)

data.average_price.fillna(0, inplace=True)
sns.distplot(data['average_price'], fit = norm)

fig = plt.figure()

res = stats.probplot(data['average_price'], plot = plt)
data.shape
sns.countplot(data['neighbourhood_group'], palette = 'plasma')

fig = plt.gcf()

fig.set_size_inches(10, 10)

plt.title('Neighbourhood Group')
sns.countplot(data['neighbourhood'], palette='plasma')

fig = plt.gcf()

fig.set_size_inches(25, 6)

plt.title('Neighbourhood')
sns.countplot(data['room_type'], palette='plasma')

fig = plt.gcf()

fig.set_size_inches(10, 10)

plt.title('Restaurants delivering online or not')
plt.figure(figsize=(10, 10))

ax = sns.boxplot(x = 'neighbourhood_group', y = 'availability_365', data=data, palette='plasma')
plt.figure(figsize=(10, 10))

sns.scatterplot(data.longitude, data.latitude, hue = data.neighbourhood_group)
plt.figure(figsize=(10, 10))

sns.scatterplot(data.longitude, data.latitude, hue=data.neighbourhood)
plt.figure(figsize=(10, 10))

sns.scatterplot(data.longitude, data.latitude, hue=data.room_type)
plt.figure(figsize=(10, 10))

sns.scatterplot(data.longitude, data.latitude, hue=data.availability_365)
m = folium.Map([40.7128,-74.0060], zoom_start=11)

HeatMap(data[['latitude', 'longitude']].dropna(), radius=8, gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1: 'red'}).add_to(m)

display(m)
data.neighbourhood_group.unique()
plt.figure(figsize=(10, 8))

sns.distplot(data[data.neighbourhood_group == 'Manhattan'].price, color='maroon', hist=False, label='Manhattan')

sns.distplot(data[data.neighbourhood_group == 'Brooklyn'].price, color='black', hist=False, label='Brooklyn')

sns.distplot(data[data.neighbourhood_group == 'Queens'].price, color='green', hist=False, label='Queens')

sns.distplot(data[data.neighbourhood_group == 'Staten Island'].price, color='blue', hist=False, label='Staten Island')

sns.distplot(data[data.neighbourhood_group == 'Bronx'].price, color='lavender', hist=False, label='Bronx')

plt.title('Borough wise price destribution for price<2000')

plt.xlim(0,2000)

plt.show()
plt.figure(figsize=(14, 8))

sns.distplot(data.minimum_nights).set_yscale('log')

plt.title('Minimum no. of nights distribution')

plt.show()
sns.set(style='white', palette='plasma', color_codes=True)

plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == 'Brooklyn'][['neighbourhood', 'price']]

d = df1.groupby('neighbourhood').mean()

sns.distplot(d, color='r', axlabel='Price distribution in Brooklyn', kde_kws={"color": "k"}, hist_kws={"histtype":"step","linewidth": 3})

plt.ioff()

plt.plot()
sns.set(style='white', palette='plasma', color_codes=True)

plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == 'Manhattan'][['neighbourhood', 'price']]

d = df1.groupby('neighbourhood').mean()

sns.distplot(d, color='r', axlabel='Price distribution in Brooklyn', kde_kws={"color": "k"}, hist_kws={"histtype":"step","linewidth": 3})

plt.ioff()

plt.plot()
sns.set(style='white', palette='plasma', color_codes=True)

plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == 'Staten Island'][['neighbourhood', 'price']]

d = df1.groupby('neighbourhood').mean()

sns.distplot(d, color='r', axlabel='Price distribution in Brooklyn', kde_kws={"color": "k"}, hist_kws={"histtype":"step","linewidth": 3})

plt.ioff()

plt.plot()
sns.set(style='white', palette='plasma', color_codes=True)

plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == 'Queens'][['neighbourhood', 'price']]

d = df1.groupby('neighbourhood').mean()

sns.distplot(d, color='r', axlabel='Price distribution in Brooklyn', kde_kws={"color": "k"}, hist_kws={"histtype":"step","linewidth": 3})

plt.ioff()

plt.plot()
sns.set(style='white', palette='plasma', color_codes=True)

plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == 'Bronx'][['neighbourhood', 'price']]

d = df1.groupby('neighbourhood').mean()

sns.distplot(d, color='r', axlabel='Price distribution in Brooklyn', kde_kws={"color": "k"}, hist_kws={"histtype":"step","linewidth": 3})

plt.ioff()

plt.plot()
plt.figure(figsize=(10, 6))

sub_6 = data[data.price < 500]

viz_4 = sub_6.plot(kind='scatter', x='longitude', y='latitude', figsize=(10, 10), colorbar=True, alpha=0.4, label='availability_365',c='price', cmap=plt.get_cmap('jet'))

viz_4.legend()

plt.ioff()
ng = data[data.price < 500]

plt.figure(figsize=(10, 6))

sns.boxplot(x='neighbourhood_group', y='price', data=ng)

plt.title('neighbourhood_group price distribution < 500')

plt.show()
df_top_prices_by_neighbourhood = data.groupby('neighbourhood').agg({'price' : 'mean'}).sort_values('price').reset_index()
# print(df_top_prices_by_neighbourhood)

df_top_prices_by_neighbourhood.columns
plt.figure(figsize=(10, 6))

sns.barplot(x = 'price', y='neighbourhood', data=df_top_prices_by_neighbourhood.head(10))

plt.ioff()
plt.figure(figsize=(10, 6))

sns.countplot(x='room_type', hue='neighbourhood_group', data=data)

plt.title('Room types occupied by neighbourhood group')

plt.show()
plt.figure(figsize=(10, 6))

sns.catplot(x='room_type', y='price', data=data)
text = " ".join(str(x) for x in data.name)

wordcloud = WordCloud(max_words=200, background_color='yellow').generate(text)

plt.figure(figsize=(10, 6))

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
_names_ = []

for name in data.name:

    _names_.append(name)

def split_name(name):

    spl = str(name).split()

    return spl

_names_for_count_ = []



for x in _names_:

        for word in split_name(x):

            word = word.lower()

            _names_for_count_.append(word)
from collections import Counter

_top_25_w = Counter(_names_for_count_).most_common()

_top_25_w = _top_25_w[0:25]
sub_w = pd.DataFrame(_top_25_w)

sub_w.rename(columns={0 : 'Words', 1 : 'Count'}, inplace=True)
plt.figure(figsize=(10, 6))

viz_5 = sns.barplot(x='Words', y='Count',data=sub_w)

viz_5.set_title('Counts of the top 25 used words for listing names')

viz_5.set_ylabel('Count of words')

viz_5.set_xlabel('Words')

viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)
plt.figure(figsize=(10, 6))

data['number_of_reviews'].plot(kind='hist')

plt.xlabel('Price')

plt.ioff()

plt.show()
df1 = data.sort_values(by=['number_of_reviews'], ascending=False).head(1000)

df1.head()
print('Rooms with most number of reviews')

Long=-73.80

Lat=40.80

mapdf1 = folium.Map([Lat, Long], zoom_start=10)

mapdf1_rooms_map = plugins.MarkerCluster().add_to(mapdf1)



for lat, lon, label in zip(df1.latitude, df1.longitude, df1.name):

    folium.Marker(location=[lat, lon], popup=label, icon=folium.Icon(icon='home')).add_to(mapdf1_rooms_map)

    

mapdf1.add_child(mapdf1_rooms_map)
data['availability_365'] = boxcox1p(data['availability_365'], 0)

data['calculated_host_listings_count'] = boxcox1p(data['calculated_host_listings_count'], 0)

data['reviews_per_month'] = boxcox1p(data['reviews_per_month'], 0)

data['number_of_reviews'] = boxcox1p(data['number_of_reviews'], 0)

data['minimum_nights'] = boxcox1p(data['minimum_nights'], 0)
data.drop(['host_id', 'id', 'name', 'host_name', 'last_review'], axis=1, inplace=True)
data_final = pd.get_dummies(data)
data_final.head()
data_final.shape
X = data_final.loc[:, data_final.columns != 'price']

y = data_final['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=50)
rf_fit = rf.fit(X_train, y_train)
y_pred_rf = rf_fit.predict(X_test)
r2_score(y_test,y_pred_rf)
from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
GBoost = GBoost.fit(X_train,y_train)
y_pred_gb = GBoost.predict(X_test)
r2_score(y_test, y_pred_gb)