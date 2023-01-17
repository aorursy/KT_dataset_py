import numpy as np

import pandas as pd 

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline
import tensorflow as tf

import sklearn

from zipfile import ZipFile
!unzip /kaggle/input/expedia-personalized-sort/data.zip
train = pd.read_csv('train.csv')
train.head(10)
train.isnull()
'''

df.plot(x='date_time', y = 'price_usd', figsize = (20,5))

plt.xlabel('Date time')

plt.ylabel('Price in USD')

plt.title('Time Series of room price by date time of search')

'''
df = train.loc[train['prop_id'] == 104517]



df = df.loc[df['visitor_location_country_id'] == 219]



df = df.loc[df['srch_room_count'] == 1]



df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
df.describe()
df = df.loc[df['price_usd'] < 5584]

df['price_usd'].describe()
print(df['date_time'].min())

print(df['date_time'].max())
df['date_time'].describe()



df['date_time'] = pd.to_datetime(df['date_time'])



df.head()
df.plot(x = 'date_time', 

        y = 'price_usd', 

        figsize = (16, 8))



plt.xlabel('dates')

plt.ylabel('USD')

plt.title('Time series of room price by date of search');
a = df.loc[df['srch_saturday_night_bool'] == 0, 'price_usd']

b = df.loc[df['srch_saturday_night_bool'] == 1, 'price_usd']



plt.figure(figsize = (16, 8))



plt.hist(a, bins = 80, 

         alpha = 0.3, 

         label = 'search w/o Sat night stay')



plt.hist(b, bins = 80, 

         alpha = 0.3, 

         label = 'search w/ Sat night stay')



plt.xlabel('Price')

plt.ylabel('Freq')

plt.legend()

plt.title('Sat night search')

plt.plot();



sns.distplot(df['price_usd'], 

                 hist = False, label = 'USD')



sns.distplot(df['srch_booking_window'], 

                  hist = False, label = 'booking window')



plt.xlabel('dist')

sns.despine()
sns.pairplot(df)
df = df.sort_values('date_time')

df['date_time_int'] = df.date_time.astype(np.int64)
sns.kdeplot(df[["price_usd", "srch_booking_window", "srch_saturday_night_bool"]])
from sklearn.cluster import KMeans

data = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

n_cluster = range(1, 20)



kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]

scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots(figsize = (16, 8))

ax.plot(n_cluster, scores, color = 'orange')



plt.xlabel('clusters num')

plt.ylabel('score')

plt.title('Elbow curve for K-Means')

plt.show();
del train
import os

os.remove('train.csv')
test = pd.read_csv('test.csv')
del kmeans
km = KMeans(n_clusters = 17).fit(data)
X = df[['price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]

X = X.reset_index(drop = True)



km.predict(X)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize = (7, 7))



ax = Axes3D(fig, rect = [0, 0, 0.95, 1], 

            elev = 48, azim = 134)



ax.scatter(X.iloc[:, 0], 

           X.iloc[:, 1], 

           X.iloc[:, 2],

           c = km.labels_.astype(np.float), edgecolor = 'm')



ax.set_xlabel('USD')

ax.set_ylabel('srch_booking_window')

ax.set_zlabel('srch_saturday_night_bool')



plt.title('K Means', fontsize = 10);
from sklearn.neural_network import BernoulliRBM

model = BernoulliRBM(n_components=2)

model.fit(data)
model.score_samples(X)
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True