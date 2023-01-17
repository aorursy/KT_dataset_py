import numpy as np

import pandas as pd

import os

import json

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

%matplotlib inline

inline_rc = dict(mpl.rcParams)
# We only use the first 100,000 data in this assignment

users = []

with open('/kaggle/input/yelp-dataset/yelp_academic_dataset_user.json') as fl:

    for i, line in enumerate(fl):

        users.append(json.loads(line))

        if i+1 >= 100000:

            break

df = pd.DataFrame(users)

df.head()
# Measures of central tendency for given data

df.describe()
# An overarching look at the missing data

# msno.matrix(df)
# Delete multiple columns from the df

df = df.drop(["user_id", "name"], axis=1)

# df.head()
# Make column friend_count = number of friends

friend_count = [0 for _ in range(df.shape[0])]

for i in range(df.shape[0]):

    friend_count[i] = len(df.loc[i, "friends"].split(","))

    

friend_count = pd.DataFrame(friend_count)

# print(friend_count)
# Add column friend count column to main db

df['friend_count'] = friend_count



# Drop column friends as not used again

df = df.drop(["friends"], axis=1)

df.head()
elite_count = [0 for _ in range(df.shape[0])]

for i in range(df.shape[0]):

    elite_count[i] = len(df.loc[i, "elite"].split(","))

    

elite_count = pd.DataFrame(elite_count)

# print(elite_count)

df['elite_count'] = elite_count  # Add column to df

df = df.drop(["elite"], axis=1) # Drop elite

df.head()
df['yelping_since'] = pd.to_datetime(df['yelping_since'])



df['yelp_since_YRMO'] = df['yelping_since'].map(lambda x: 100*x.year + x.month)

df['yelp_since_year'] = df['yelping_since'].dt.year



df.head()
# Column to store whether compliment has been tagged

tagged_compliment = [0 for _ in range(df.shape[0])]

for i in range(df.shape[0]):

    if sum(df.iloc[i, 7:18].values) > 0:

        tagged_compliment[i] = 1

        

tagged_compliment = pd.DataFrame(tagged_compliment)

df['tagged_compliment'] = tagged_compliment
# Plot count vs yearmonth, to see the distribution

plt.figure(figsize=(18,4))

yrmo = pd.to_datetime(df['yelp_since_YRMO'], format='%Y%m')

yrmo = pd.DataFrame(yrmo)

yrmo.yelp_since_YRMO.value_counts().plot(kind='line')
plt.figure(figsize=(12,2))

year = pd.to_datetime(df['yelp_since_year'], format='%Y')

year = pd.DataFrame(year)

year.yelp_since_year.value_counts().plot(kind='line')
# Time Period 201201-201212 | 201301-201312 | 201401-201412

plt.figure(figsize=(20,2))

period_12 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201201, format='%Y%m')]

period_12 = period_12[period_12.yelp_since_YRMO <= pd.to_datetime(201212, format='%Y%m')]

period_12 = pd.to_datetime(period_12.yelp_since_YRMO, format='%Y%m')



period_13 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201301, format='%Y%m')]

period_13 = period_13[period_13.yelp_since_YRMO <= pd.to_datetime(201312, format='%Y%m')]

period_13 = pd.to_datetime(period_13.yelp_since_YRMO, format='%Y%m')



period_14 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201401, format='%Y%m')]

period_14 = period_14[period_14.yelp_since_YRMO <= pd.to_datetime(201412, format='%Y%m')]

period_14 = pd.to_datetime(period_14.yelp_since_YRMO, format='%Y%m')



plt.subplot(131)

period_12.value_counts().plot(kind='line', linewidth=2, color='b')

plt.subplot(132)

period_13.value_counts().plot(kind='line', linewidth=2, color='b')

plt.subplot(133)

period_14.value_counts().plot(kind='line', linewidth=2, color='b')

plt.show()
# Time Period 201501-201512 | 201601-201612 | 201701-201712

plt.figure(figsize=(20,2))

period_15 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201501, format='%Y%m')]

period_15 = period_15[period_15.yelp_since_YRMO <= pd.to_datetime(201512, format='%Y%m')]

period_15 = pd.to_datetime(period_15.yelp_since_YRMO, format='%Y%m')



period_16 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201601, format='%Y%m')]

period_16 = period_16[period_16.yelp_since_YRMO <= pd.to_datetime(201612, format='%Y%m')]

period_16 = pd.to_datetime(period_16.yelp_since_YRMO, format='%Y%m')



period_17 = yrmo[yrmo.yelp_since_YRMO >= pd.to_datetime(201701, format='%Y%m')]

period_17 = period_17[period_17.yelp_since_YRMO <= pd.to_datetime(201712, format='%Y%m')]

period_17 = pd.to_datetime(period_17.yelp_since_YRMO, format='%Y%m')



plt.subplot(131)

period_15.value_counts().plot(kind='line', linewidth=2, color='b')

plt.subplot(132)

period_16.value_counts().plot(kind='line', linewidth=2, color='b')

plt.subplot(133)

period_17.value_counts().plot(kind='line', linewidth=2, color='b')

plt.show()
# Drop yelping_since column from df as not used again, and we already store lower granularity data in year & yrmo

df = df.drop(["yelping_since"], axis=1)
plt.figure(figsize=(16,3))

sns.distplot(df.average_stars)
raters_below_3 = len(df.loc[df.average_stars <= 3])

print("Users who rate <= 3 Avg Stars: {:0.02%}".format(raters_below_3/df.shape[0]))
low_raters = len(df.loc[df.average_stars < 4])

high_raters = len(df.loc[df.average_stars >= 4])

print("Low Raters, <4 Avg Stars: {:0.02%}".format(low_raters/df.shape[0]))

print("High Raters >=4 Avg Stars: {:0.02%}".format(high_raters/df.shape[0]))
# Make a column raters, which is 1 for high raters (>=4 avg stars), and 0 for the rest (<4)



raters = [0 for _ in range(df.shape[0])]

for i in range(df.shape[0]):

    if df.loc[i,"average_stars"] >= 4:

        raters[i] = 1

#     elif float(3) <= df.loc[i,"average_stars"] < float(4):

#         rating[i] = "M"

#     else:

#         raters[i] = "H"

# Add column to main df

df['raters'] = raters
plt.figure(figsize=(16,3))

plt.subplot(121)

sns.distplot(df.review_count)



# Taking a Normal Distribution as review_count heavily skewed

plt.subplot(122)

sns.distplot(df.review_count.apply(np.log1p))
plt.figure(figsize=(16,3))

plt.subplot(121)

sns.distplot(df.friend_count)



# Taking a Normal Distribution as friend_count heavily skewed

plt.subplot(122)

sns.distplot(df.friend_count.apply(np.log1p))
plt.figure(figsize=(24,3))



plt.subplot(141)

sns.distplot(df.fans)



# Taking a Normal Distribution as fans heavily skewed

plt.subplot(142)

sns.distplot(df.fans.apply(np.log1p))



plt.subplot(143)

sns.distplot(df.elite_count)



# Taking a Normal Distribution as elite_count heavily skewed

plt.subplot(144)

sns.distplot(df.elite_count.apply(np.log1p))
useful_reviews = len(df.loc[df.useful > 0])

print("People who leave useful reviews: {:0.0%}".format(useful_reviews/df.shape[0]))
plt.figure(figsize=(16,3))

plt.subplot(121)

sns.distplot(df.useful)



# Taking a Normal Distribution as useful count heavily skewed

plt.subplot(122)

sns.distplot(df.useful.apply(np.log1p))
from sklearn.preprocessing import StandardScaler



# Don't scale columns: yelp_since_year, yelp_since_YRMO, elite

features = ['review_count', 'useful', 'funny', 'cool', 'fans',

       'average_stars', 'compliment_hot', 'compliment_more',

       'compliment_profile', 'compliment_cute', 'compliment_list',

       'compliment_note', 'compliment_plain', 'compliment_cool',

       'compliment_funny', 'compliment_writer', 'compliment_photos',

       'friend_count', 'elite_count', 'raters', 'tagged_compliment']

x = df.loc[:, features]

x = StandardScaler().fit_transform(x)
# Adding column names back to data, and converting ndarray back to datafram obj

df_train = pd.DataFrame(x, columns=features)

df_train.head()
# Eyebaling central tendency, and qc-ing scaled values

df_train.describe()
# Covariance matrix of scaled data

cov = df_train.cov()

cov.style.background_gradient(cmap='coolwarm').set_precision(2)
# Correlation matrix of unscaled original data which includes, elite count, yrmo features etc

# Idea is to see which features are correlated and can be combined (PCA) together to build a hypothesis/cluster



corr = df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
df_train.columns
df_review = df_train.loc[:, ['funny', 'cool', 'useful']]  # Make df of these 3 features

pca = PCA(n_components=1)

review_feedback = pca.fit_transform(df_review)

review_feedback = pd.DataFrame(data=review_feedback)
print('PCA Components:', pca.components_)

print('Ratio of Variance Explained:', pca.explained_variance_ratio_ )
df_compliments = df_train.loc[:, ['compliment_hot', 'compliment_more', 'compliment_profile',

       'compliment_cute', 'compliment_list', 'compliment_note',

       'compliment_plain', 'compliment_cool', 'compliment_funny',

       'compliment_writer', 'compliment_photos']]

pca = PCA(n_components=1)

compliments_feedback = pca.fit_transform(df_compliments)

compliments_feedback = pd.DataFrame(data=compliments_feedback)
print('PCA Components:', pca.components_)

print('Ratio of Variance Explained:', pca.explained_variance_ratio_ )
df_popularity = df_train.loc[:, ['fans', 'friend_count']]

pca = PCA(n_components=1)

popularity_feedback = pca.fit_transform(df_popularity)

popularity_feedback = pd.DataFrame(data=popularity_feedback)
print('PCA Components:', pca.components_)

print('Ratio of Variance Explained:', pca.explained_variance_ratio_ )
df_active = df_train.loc[:, ['review_count', 'elite_count']]

pca4 = PCA(n_components=1)

active_feedback = pca4.fit_transform(df_active)

active_feedback = pd.DataFrame(data=active_feedback)
print('PCA Components:', pca.components_)

print('Ratio of Variance Explained:', pca.explained_variance_ratio_ )
from yellowbrick.cluster import KElbowVisualizer

comp_star = pd.concat([compliments_feedback, df.loc[:,'average_stars']], axis=1)

model = KElbowVisualizer(KMeans(), k=10, metric='calinski_harabasz', timings=False)

model.fit(comp_star)

model.show()
# Reset matplotlib parameters, changed by elbow visualizer

mpl.rcParams.update(mpl.rcParamsDefault)
model = KMeans(n_clusters=3)

model.fit(comp_star)

all_predictions = model.predict(comp_star)

centroids = model.cluster_centers_



plt.figure(figsize=(14, 3))

plt.scatter(comp_star.iloc[:,0].values, comp_star.iloc[:,1].values, c=all_predictions)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')

plt.xlabel('Compliments Feedback')

plt.ylabel('Average Stars')

plt.show()
act_star = pd.concat([active_feedback, df.loc[:,'average_stars']], axis=1)

model = KMeans(n_clusters=2)

model.fit(act_star)

all_predictions = model.predict(act_star)

centroids = model.cluster_centers_



plt.figure(figsize=(14, 3))

plt.scatter(act_star.iloc[:,0].values, act_star.iloc[:,1].values, c=all_predictions)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')

plt.xlabel('Active Feedback (Review Count, Elite Status Count)')

plt.ylabel('Average Stars')

plt.show()
pop_stars = pd.concat([popularity_feedback, df.loc[:,'average_stars']], axis=1)

model = KMeans(n_clusters=3)

model.fit(pop_stars)

all_predictions = model.predict(pop_stars)

centroids = model.cluster_centers_



plt.figure(figsize=(14, 3))

plt.scatter(pop_stars.iloc[:,0].values, pop_stars.iloc[:,1].values, c=all_predictions)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')

plt.xlabel('Popularity Feedback (Friends, Fans)')

plt.ylabel('Average Stars')

plt.show()
rev_pop = pd.concat([review_feedback, popularity_feedback], axis=1)

print("Correlation: {:0.02%}".format((rev_pop.corr()).iloc[0,1]))
model = KMeans(n_clusters=3)

model.fit(rev_pop)

all_predictions = model.predict(rev_pop)

centroids = model.cluster_centers_



plt.figure(figsize=(14, 3))

plt.scatter(rev_pop.iloc[:,0].values, rev_pop.iloc[:,1].values, c=all_predictions)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')

plt.xlabel('Review Feedback (useful, funny, cool)')

plt.ylabel('Popularity Feedback (fans, friends)')

plt.show()
rev_act = pd.concat([review_feedback, active_feedback], axis=1)

print("Correlation: {:0.02%}".format((rev_act.corr()).iloc[0,1]))
model = KMeans(n_clusters=4)

model.fit(rev_act)

all_predictions = model.predict(rev_act)

centroids = model.cluster_centers_



plt.figure(figsize=(12,4))

plt.scatter(rev_act.iloc[:,0].values, rev_act.iloc[:,1].values, c=all_predictions)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')

plt.xlabel('Review Feedback (useful, funny, cool)')

plt.ylabel('Active Feedback (review, elite counts)')