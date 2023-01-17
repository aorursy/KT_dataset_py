from collections import defaultdict

from folium import plugins, Icon, Map, Marker, Circle

from folium.plugins import MarkerCluster, HeatMap

from geopy import distance

from matplotlib import cm

from matplotlib import colors as pltcolors

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import numpy as np

import pandas as pd

from random import sample

import scipy.stats as ss

import seaborn as sns

from sklearn import metrics

from sklearn.cluster import DBSCAN, OPTICS

from sklearn.preprocessing import RobustScaler

%matplotlib inline

plt.style.use('seaborn-dark')
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
print("Number of rows:%d" % df.shape[0])

print("Number of cols:%d" % df.shape[1])
df.info()
df['last_review'] = pd.to_datetime(df['last_review'])
df.nunique()
df.isna().sum()
df[df['number_of_reviews'] == 0 & df['last_review'].isna() & df['reviews_per_month'].isna()].shape[0]
df['reviews_per_month'].fillna(0, inplace=True)

data = df.drop(['last_review'], axis=1)
names_count = df['name'].value_counts()

print("There's a total of %d unique names" % names_count.gt(1).sum())

names_count.head(10)
names_count = df['host_name'].value_counts()

print("There's a total of %d unique host names" % names_count.gt(1).sum())

names_count.head(10)
data.drop(['name', 'host_name'], inplace=True, axis=1)
data.describe(include='all')
cols = np.concatenate([data.columns.values[2:4], data.columns.values[6:]])

corr_matrix = data[cols].corr()

plt.figure(figsize=(10, 6))

g = sns.heatmap(corr_matrix, annot=True,

        xticklabels=corr_matrix.columns,

        yticklabels=corr_matrix.columns)
def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



cramers_dict = defaultdict(lambda: defaultdict())

cols_len = len(cols)

for i, coli in enumerate(cols):

    cramers_dict[coli][coli] = 1

    for j in range(i+1, cols_len):

        colj = cols[j]

        c = cramers_v(data[coli], data[colj])

        cramers_dict[coli][colj] = c

        cramers_dict[colj][coli] = c

df_cramers = pd.DataFrame.from_dict(cramers_dict)

plt.figure(figsize=(12, 8))

ax = sns.heatmap(df_cramers, annot=True)
import math

from collections import Counter

import scipy.stats as ss



def conditional_entropy(x,y):

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def theil_u(x,y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x

    

theil_dict = defaultdict(lambda: defaultdict())

cols_len = len(cols)

for i, coli in enumerate(cols):

    theil_dict[coli][coli] = 1

    for j in range(i+1, cols_len):

        colj = cols[j]

        c = theil_u(data[coli], data[colj])

        theil_dict[coli][colj] = c

        theil_dict[colj][coli] = c

df_theil = pd.DataFrame.from_dict(theil_dict)

plt.figure(figsize=(12, 8))

ax = sns.heatmap(df_theil, annot=True)
g = sns.pairplot(data.drop(['id', 'host_id', 'neighbourhood', 'latitude', 'longitude'], axis=1), hue="neighbourhood_group")
pd.crosstab(data['room_type'], data['neighbourhood_group'], margins=True)

pd.pivot_table(data, 

               index=['neighbourhood_group', 'room_type'], 

               values=['price', 'number_of_reviews', 'reviews_per_month', 'minimum_nights', 'availability_365'],

               aggfunc=np.mean).round(2)
g = pd.crosstab(data['neighbourhood_group'], data['room_type']).plot.bar(figsize=(12,8))
pd.crosstab(data['room_type'], data['neighbourhood_group'], values=data['price'], aggfunc='mean')
g = pd.crosstab(data['neighbourhood_group'], data['room_type'], values=data['price'], aggfunc='mean').plot.bar(figsize=(12,8))
top = 10

cols = 2

rows = 3

ngroups = data['neighbourhood_group'].unique()

all_colors = [k for k,v in pltcolors.cnames.items()]



def random_colors(n_items: int):

    return sample(all_colors, n_items)



def plot_neighboorhood_pie(ntotals, ng, i):

    plt.subplot(3, 2, i+1)

    labels = [''] * (top + 1)

    ax = ntotals.plot.pie(autopct='%1.1f%%', labels=labels, colors=random_colors(top + 1))

    plt.legend(labels=ntotals.index, loc=0, bbox_to_anchor=(1,0.75), title="Top neighboorhoods in %s" % ng)

    l = plt.xlabel(ng, fontsize=18)

    plt.ylabel('')



def get_totals(ng):

    totals = data[data['neighbourhood_group'] == ng]['neighbourhood'].value_counts()

    top_results = totals[:top]

    others = totals[top:]

    top_results['Other neighborhoods (%d)' % (len(others))] = sum(others)

    return top_results

     

fig = plt.figure(figsize=(20, 25))

fig.subplots_adjust()

for i, ng in enumerate(ngroups):

    totals = get_totals(ng)

    plot_neighboorhood_pie(totals, ng, i)
def get_center(points):

    """

    Given a dataframe, calculate and return the center.

    """

    return[points['latitude'].mean(), points['longitude'].mean()]



# Default zoom for the maps.

zoom = 12



# When displaying more markers than this threshold, start using clusters for grouping.

cluster_threshold = 50





def icon_for_room(room_type) -> Icon:

    """

    Create an icon based on the room type. Using different icons help distinguishing the

    different places by their room type.

    """

    if room_type == 'Entire home/apt':

        return Icon(icon='home', color='purple')

    if room_type == 'Private room':

        return Icon(icon='building', prefix='fa', color='green')

    return Icon(icon='bed', prefix='fa')





def create_heatmap(points):

    """

    Create a heatmap for the given points.

    """

    hmap = Map(location=get_center(points), zoom_start=zoom)

    llpairs = [[latitude, longitude] for latitude, longitude in zip(points['latitude'], points['longitude'])]

    hmap.add_child(HeatMap(llpairs))

    return hmap





def plot_points_within_radius(points, center, label, radius):

    """

    Plot all the points within a radius from a center.

    """

    radius = radius * 1609

    pmap = Map(center, zoom_start=zoom)

    pmap.add_child(Circle([center[0], center[1]], radius=radius, popup=label, fill_color='#d6e9ff', fill_opacity=0.6))

    zpoints = zip(points['latitude'], points['longitude'], points['name'], points['price'], points['room_type'])

    for latitude, longitude, name, price, room_type in zpoints:

        point = Marker(popup="%s ($%.2f)" % (name, price), location=[latitude, longitude], icon=icon_for_room(room_type))

        pmap.add_child(point)

    pmap.add_child(Marker(popup=label, location=[center[0], center[1]], icon=Icon(icon='university', prefix='fa', color='darkblue')))

    return pmap





def plot_in_map(points):

    """

    Plot the given points in a map with a different icon depending on the room type.

    """

    center = get_center(points)

    pmap = Map(center, zoom_start=zoom)

    container = pmap

    if len(points) > cluster_threshold:

        cluster = MarkerCluster()

        pmap.add_child(cluster)

        container = cluster

    zpoints = zip(points['latitude'], points['longitude'], points['name'], points['price'], points['room_type'])

    for latitude, longitude, name, price, room_type in zpoints:

        point = Marker(popup="%s ($%.2f)" % (name, price), location=[latitude, longitude], icon=icon_for_room(room_type))

        container.add_child(point)

    return pmap
top50 = df[(df['availability_365'] == 365) & (df['neighbourhood_group'] == 'Manhattan')].sort_values(by='price').iloc[:50]

plot_in_map(top50)
shared_rooms_one_night = data[(data['room_type'] == 'Shared room') & (data['minimum_nights'] == 1)]

review_per_month_mean = shared_rooms_one_night['reviews_per_month'].mean()



points = shared_rooms_one_night[shared_rooms_one_night['reviews_per_month'] > review_per_month_mean]



create_heatmap(points)
madison_square_garden = (40.750298, -73.993324)



def closer_than_distance(point_a, point_b, radius):

    return distance.distance(point_a, point_b).mi < radius



def points_within_radius(points, center, radius):

    return points[points.apply(lambda p: closer_than_distance(center, (p['latitude'], p['longitude']), radius), axis=1)]





points = df[(df['price'] < 75)]

radius = 1

points_in_radius = points_within_radius(points, madison_square_garden, radius)



plot_points_within_radius(points_in_radius, madison_square_garden, "Madison Square Garden", radius)
ocols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']

data[ocols].quantile([.25, .5, .75, .98, .99, 1])
def plot_densities_and_get_outliers(col: str, threshold: int):

    """

    This function display the boxplot, violin plot, density and z-core

    for the selected feature, returning those places where the z-score

    is larger than a given threshold.

    """

    fig = plt.figure(figsize=(16, 16))

    fig.subplots_adjust()

    df_col = df[col]

    plt.subplot(2, 2, 1)

    sns.boxplot(x = df_col)

    plt.subplot(2, 2, 2)

    ax = df_col.hist()

    ax.set_xlabel(col)

    z = np.abs(ss.stats.zscore(df_col))

    plt.subplot(2, 2, 3)

    ax = pd.Series(z).plot.kde()

    ax.set_xlabel("z-score %s" % col)

    plt.subplot(2, 2, 4)

    ax = sns.violinplot(df_col)

    return data.iloc[np.where(z > threshold)].sort_values(by=col, ascending=False)
plot_densities_and_get_outliers('price', 3)
quantiles = data['price'].quantile([.98, .99])

for i, q in quantiles.items():

    print('Places above the %dth quantile (%.2f): %d' % (i*100, q, data[data['price'] > q].shape[0]))
plot_densities_and_get_outliers('minimum_nights', 3)
plot_densities_and_get_outliers('number_of_reviews', 3)
plot_densities_and_get_outliers('reviews_per_month', 3)
plot_densities_and_get_outliers('calculated_host_listings_count', 3)
"""

min_pts = int(np.log(data.shape[0]))

epsilon = 0.75

prices = data['price']



rs_prices = RobustScaler().fit_transform(prices.values.reshape(-1, 1))

db = DBSCAN(min_samples=min_pts, eps=epsilon).fit(rs_prices)

bool_labels = [True if l == -1 else False for l in db.labels_]

price_outliers = df[bool_labels]

"""
"""

clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print('Number of clusters: %d' % clusters)

print('Number of outliers: %d' % price_outliers.shape[0])

price_outliers.head()

"""
"""

with np.errstate(divide='ignore'):

    clustering = OPTICS(min_samples=min_pts, eps=epsilon).fit(rs_prices)

bool_labels = [True if l == -1 else False for l in clustering.labels_]

price_outliers = df[bool_labels]

print('Number of outliers: %d' % price_outliers.shape[0])

price_outliers.head()

"""