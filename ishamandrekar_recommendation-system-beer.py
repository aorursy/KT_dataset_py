# import libraties

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

# Reading ratings file

ratings = pd.read_csv('../input/beer_data.csv')
ratings.head()
# Checking the percentage of missing values

col_list = ratings.columns



for col_name in ratings.columns:

    missing_percent = round(100* ((ratings[col_name].isnull()) | (ratings[col_name].astype(str) == 'Select')).sum() /len(ratings.index) , 2)

    print(col_name + " - " + str(missing_percent))
ratings.beer_beerid.nunique()
ratings.review_profilename.nunique()
ratings.review_overall.nunique()
ratings.review_overall.value_counts()
ratings.loc[ratings.review_profilename.isnull()].count()
ratings.count()
(ratings.loc[ratings.review_profilename.isnull()].beer_beerid.count()*100)/ratings.beer_beerid.count()
ratings = ratings[~ratings.review_profilename.isnull()]
ratings.count()
ratings.duplicated().sum()
ratings[(ratings.beer_beerid == 73647) & (ratings.review_profilename == 'barleywinefiend') & (ratings.review_overall==4.5)]
ratings.drop_duplicates(keep = 'first', inplace = True)
ratings.duplicated().sum()
ratings[(ratings.beer_beerid == 73647) & (ratings.review_profilename == 'barleywinefiend') & (ratings.review_overall==4.5)]
ratings.count()
review_count_by_beer_id = ratings.groupby('beer_beerid').review_overall.count().to_frame('count_reviews').sort_values(by = "count_reviews", ascending = False)
review_count_by_beer_id.reset_index(inplace=True)
review_count_by_beer_id.head()
review_count_by_beer_id.count()
review_count_analyze = review_count_by_beer_id.count_reviews.value_counts().to_frame().reset_index()
review_count_analyze.columns = ['no_of_reviews','How_many_such_beer_ids']
review_count_analyze.count()
review_count_analyze['percentage_beers'] = (review_count_analyze['How_many_such_beer_ids']*100)/review_count_analyze.How_many_such_beer_ids.sum()

review_count_analyze['cumulative_percentage_beers'] = review_count_analyze.percentage_beers.cumsum()
review_count_analyze.head(10)
sns.set(rc={'figure.figsize':(25,7)})

ax = sns.barplot(x=review_count_analyze[0:100].no_of_reviews, y=review_count_analyze[0:200].How_many_such_beer_ids, palette="rocket")

ax.set(yscale="log")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10)

ax
review_count_by_profilename = ratings.groupby('review_profilename').review_overall.count().to_frame('count_reviews').sort_values(by = "count_reviews", ascending = False)
review_count_by_profilename.reset_index(inplace=True)
review_count_by_profilename.head()
review_count_by_profilename.count()
review_count_user_analyze = review_count_by_profilename.count_reviews.value_counts().to_frame().reset_index()
review_count_user_analyze.columns = ['no_of_reviews','How_many_such_users']
review_count_user_analyze.count()
review_count_user_analyze['percentage_users'] = (review_count_user_analyze['How_many_such_users']*100)/review_count_user_analyze.How_many_such_users.sum()

review_count_user_analyze['cumulative_percentage_users'] = review_count_user_analyze.percentage_users.cumsum()
review_count_user_analyze.head(10)
sns.set(rc={'figure.figsize':(25,7)})

ax = sns.barplot(x=review_count_user_analyze[0:100].no_of_reviews, y=review_count_user_analyze[0:200].How_many_such_users, palette="rocket")

ax.set(yscale="log")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10)

ax
ratings.beer_beerid.nunique()
ratings.review_profilename.nunique()
ratings.beer_beerid.count()
beer_ids_no_of_ratings_grte_20 = review_count_by_beer_id.loc[review_count_by_beer_id.count_reviews>=20].beer_beerid.to_frame('beer_beerid')

users_no_of_ratings_grte_20 = review_count_by_profilename.loc[review_count_by_profilename.count_reviews>=20].review_profilename.to_frame('review_profilename')
ratings1 = pd.merge(ratings, beer_ids_no_of_ratings_grte_20, how='inner', on='beer_beerid')

ratings1 = pd.merge(ratings1, users_no_of_ratings_grte_20, how='inner', on='review_profilename')
ratings1.head()
ratings1.count()
np.sort(ratings1.review_overall.unique())
ratings1.head()
ratings1.groupby('beer_beerid').review_overall.mean()
#sns.distplot(ratings1.groupby('beer_beerid').review_overall.mean(), kde=False, color="b")

sns.distplot(ratings1.groupby('beer_beerid').review_overall.mean(), color="m")
ratings1.groupby('beer_beerid').review_overall.mean().mean()
ratings1.groupby('review_profilename').review_overall.mean()
sns.distplot(ratings1.groupby('review_profilename').review_overall.mean(), color="m")
ratings1.groupby('review_profilename').review_overall.mean().mean()
review_count_by_beer_id1 = ratings1.groupby('beer_beerid').review_overall.count().to_frame('count_reviews').sort_values(by = "count_reviews", ascending = False)
review_count_by_beer_id1.head()
sns.distplot(review_count_by_beer_id1, color="m")
review_count_by_beer_id1.count_reviews.mean()
review_count_by_user = ratings1.groupby('review_profilename').review_overall.count().to_frame('count_reviews').sort_values(by = "count_reviews", ascending = False)
review_count_by_user.head()
sns.distplot(review_count_by_user, color="m")
review_count_by_user.count_reviews.mean()
from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings1, test_size=0.30, random_state=31)
print(train.shape)

print(test.shape)
train.head()
train.reset_index(inplace=True,drop=True)

test.reset_index(inplace=True,drop=True)
train.head()
# pivot ratings beer features

df_beer_features = train.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

).fillna(0)
df_beer_features.head()
dummy_train = train.copy()

dummy_test = test.copy()
dummy_train['review_overall'] = dummy_train['review_overall'].apply(lambda x: 0 if x>=1 else 1)

dummy_test['review_overall'] = dummy_test['review_overall'].apply(lambda x: 1 if x>=1 else 0)
# The beers not rated by user is marked as 1 for prediction. 

dummy_train = dummy_train.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

).fillna(1)



# The beers not rated by user is marked as 0 for evaluation. 

dummy_test = dummy_test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

).fillna(0)
dummy_train.head()
dummy_test.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = 1 - pairwise_distances(df_beer_features, metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation.shape
beer_features = train.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

)
beer_features.head()
mean = np.nanmean(beer_features, axis=1)

df_subtracted = (beer_features.T-mean).T
df_subtracted.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation[user_correlation<0]=0

user_correlation
user_predicted_ratings = np.dot(user_correlation, beer_features.fillna(0))

user_predicted_ratings
user_predicted_ratings.shape
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

user_final_rating.head()
user_final_rating.iloc[0].sort_values(ascending=False)[0:5]
beer_features = train.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

).T



beer_features.head()
mean = np.nanmean(beer_features, axis=1)

df_subtracted1 = (beer_features.T-mean).T
df_subtracted1.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

item_correlation = 1 - pairwise_distances(df_subtracted1.fillna(0), metric='cosine')

item_correlation[np.isnan(item_correlation)] = 0

print(item_correlation)
item_correlation[item_correlation<0]=0

item_correlation
item_predicted_ratings = np.dot((beer_features.fillna(0).T),item_correlation)

item_predicted_ratings
item_predicted_ratings.shape
dummy_train.shape
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)

item_final_rating.head()
item_final_rating.iloc[0].sort_values(ascending=False)[0:5]
user_correlation
user_correlation.shape
user_correlation1= user_correlation
np.fill_diagonal(user_correlation1, 0)
def largest_indices(ary, n):

    """Returns the n largest indices from a numpy array."""

    flat = ary.flatten()

    indices = np.argpartition(flat, -n)[-n:]

    indices = indices[np.argsort(-flat[indices])]

    return np.unravel_index(indices, ary.shape)
a = largest_indices(user_correlation1, 20)

a
user_correlation1[largest_indices(user_correlation1, 20)]
for i in list(a[0]):

     print(str(i) + "-" + beer_features.columns.get_values()[i])
largest_indices(user_correlation1, 20)
sns.set(rc={'figure.figsize':(20,20)})

sns.heatmap(df_subtracted.iloc[:,list(a[0])].dropna(thresh=1),cmap='RdBu')
item_correlation
item_correlation.shape
item_correlation1 = item_correlation
np.fill_diagonal(item_correlation1, 0)
b = largest_indices(item_correlation1, 20)

b
item_correlation1[largest_indices(item_correlation1, 20)]
for i in list(b[0]):

   print(str(i) + "-" + str(beer_features.T.columns.get_values()[i]))
sns.set(rc={'figure.figsize':(20,20)})

sns.heatmap(df_subtracted1.iloc[list(b[0]),:].dropna(thresh=1).T,cmap='RdBu')
test.head()
test.info()
test_beer_features = test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

)

mean = np.nanmean(test_beer_features, axis=1)

test_df_subtracted = (test_beer_features.T-mean).T



# User Similarity Matrix

test_user_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_user_correlation[np.isnan(test_user_correlation)] = 0

print(test_user_correlation)
test_user_correlation[test_user_correlation<0]=0

test_user_predicted_ratings = np.dot(test_user_correlation, test_beer_features.fillna(0))

test_user_predicted_ratings
test_user_final_rating = np.multiply(test_user_predicted_ratings,dummy_test)
test_user_final_rating.head()
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_user_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))



print(y)
test_ = test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

)
# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)
test_beer_features = test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

).T



mean = np.nanmean(test_beer_features, axis=1)

test_df_subtracted = (test_beer_features.T-mean).T



test_item_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_item_correlation[np.isnan(test_item_correlation)] = 0

test_item_correlation[test_item_correlation<0]=0
test_item_correlation.shape
test_beer_features.shape
test_item_predicted_ratings = (np.dot(test_item_correlation, test_beer_features.fillna(0))).T

test_item_final_rating = np.multiply(test_item_predicted_ratings,dummy_test)

test_item_final_rating.head()
test_ = test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

)
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_item_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))





test_ = test.pivot_table(

    index='review_profilename',

    columns='beer_beerid',

    values='review_overall'

)



# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)
#Top 5 beer recommendations to Cokes using User Collaborative Filtering Model

user_final_rating.loc['cokes',:].sort_values(ascending=False)[0:5]
#Top 5 beer recommendations to genog using User Collaborative Filtering Model

user_final_rating.loc['genog',:].sort_values(ascending=False)[0:5]
#Top 5 beer recommendations to giblet using User Collaborative Filtering Model

user_final_rating.loc['giblet',:].sort_values(ascending=False)[0:5]
#Top 5 beer recommendations to Cokes using Item Collaborative Filtering Model

item_final_rating.loc['cokes',:].sort_values(ascending=False)[0:5]
#Top 5 beer recommendations to genog using Item Collaborative Filtering Model

item_final_rating.loc['genog',:].sort_values(ascending=False)[0:5]
#Top 5 beer recommendations to giblet using Item Collaborative Filtering Model

item_final_rating.loc['giblet',:].sort_values(ascending=False)[0:5]