import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import missingno

%matplotlib inline

%config inlinebackend.figure_format = 'retina' 



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.model_selection import train_test_split

from sklearn import model_selection, preprocessing, metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

import scipy.cluster.hierarchy as sch



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
bnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# View the air bnn data

bnb.head()
bnb.info()
bnb.describe().columns # show only columns which is numeric
bnb['neighbourhood_code'] = bnb.groupby(pd.Grouper(key='neighbourhood')).ngroup()

bnb['neighbourhood_group_code'] = bnb.groupby(pd.Grouper(key='neighbourhood_group')).ngroup()

bnb['room_type_code'] = bnb.groupby(pd.Grouper(key='room_type')).ngroup()

#bnb['price_inv'] = bnb.price.max()-bnb.price
#cols = ['latitude', 'longitude', 'price', 'minimum_nights',

#       'number_of_reviews', 'reviews_per_month',

#       'calculated_host_listings_count', 'availability_365','neighbourhood_code','neighbourhood_group_code','room_type_code']

cln = ['latitude', 'longitude', 'price', 'minimum_nights',

       'number_of_reviews', 'reviews_per_month',

       'calculated_host_listings_count', 'availability_365','room_type_code']

cols = ['latitude', 'longitude', 'price']

#cols = ['price', 'availability_365']

#linkage = ['ward', 'complete', 'average', 'single']

linkage = ['complete']
# hierarchical clustering complete linkage

X = bnb[cln].sample(10000)

X = X.fillna(0) # replace nan of review per month to zero

X = MinMaxScaler().fit_transform(X)



agg = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage ='complete')

y_agg = agg.fit_predict(X)



df = pd.DataFrame(X, columns=cln)

df['Group'] = y_agg

df
# dendrogram แสดงช่วง Euclidean ที่ห่างๆกัน เลือกลากเส้นแนวนอนในชั้นที่3ได้ 8กลุ่ม

plt.figure(figsize=(30,5))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))

plt.title(f'Dendrogram linkage: complete')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
# hierarchical clustering complete linkage แบ่งข้อมูลเป็น 3 cluser จาก latitude, longitude, price

cols = ['latitude', 'longitude', 'price']

X = bnb[cols].sample(10000)

X = X.fillna(0) # replace nan of review per month to zero

X = MinMaxScaler().fit_transform(X)

agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage ='complete')

y_agg = agg.fit_predict(X)



df = pd.DataFrame(X, columns=cols)

df['Group'] = y_agg

df
# Plot

fig = plt.figure(figsize=(6,6))

ax = plt.axes(projection='3d')

ax.scatter(df.iloc[:,0], df.iloc[:, 1], df.iloc[:, 2], c=df['Group'])

ax.view_init(12, -60)
sns.lmplot(data=df, x='latitude', y='longitude', hue='Group', palette='deep',fit_reg=False,scatter_kws={'alpha':0.5})
sns.lmplot(data=df, x='latitude', y='price', hue='Group', palette='deep',fit_reg=False,scatter_kws={'alpha':0.5})
sns.lmplot(data=df, x='longitude', y='price', hue='Group', palette='deep',fit_reg=False,scatter_kws={'alpha':0.5})
# dendrogram แสดงช่วง Euclidean ที่ห่างๆกัน เลือกลากเส้นแนวนอนในชั้นที่2ได้ 3กลุ่ม

plt.figure(figsize=(30,5))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'complete'))

plt.title(f'Dendrogram linkage: complete')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()