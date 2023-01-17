import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/beer-datacsv/beer_data.csv', encoding='iso-8859-1')
df.head()
#checking % of missing values
df.isnull().sum() * 100 / len(df)
df.count()
#checking review count for each rating
df.review_overall.value_counts()
#Checking the review_profilename null rows
df.loc[df.review_profilename.isnull()].count()
#since very small no. of rows for null values so removing those
df = df[~df.review_profilename.isnull()]

df.isnull().sum() * 100 / len(df)
df.duplicated().sum()
df.drop_duplicates(keep = 'first', inplace = True)
df.duplicated().sum()
reviews_count_beer_id = df.groupby('beer_beerid').review_overall.count().to_frame('Reviews_count').sort_values(by = "Reviews_count", ascending=False)
reviews_count_beer_id.head()
reviews_count_beer_id.reset_index(inplace=True)
reviews_count_beer_id.head()
reviews_count_analyze = reviews_count_beer_id.Reviews_count.value_counts().to_frame().reset_index()
reviews_count_analyze.columns = ['number_of_reviews','number_of_beer_ids']
reviews_count_analyze.head()
#calculating percentage of each beer_id and also their cummulative percentage
reviews_count_analyze['percentage_beers'] = (reviews_count_analyze['number_of_beer_ids']*100)/reviews_count_analyze.number_of_beer_ids.sum()
reviews_count_analyze['cumulative_percentage_beers'] = reviews_count_analyze.percentage_beers.cumsum()
reviews_count_analyze.head(10)
reviews_count_analyze.count()
import seaborn as sns

sns.set(rc={'figure.figsize':(25,7)})
ax = sns.barplot(x=reviews_count_analyze[0:100].number_of_reviews, y=reviews_count_analyze[0:200].number_of_beer_ids, palette="rocket")
ax.set(yscale="log")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10)
ax
reviews_count_by_profilename = df.groupby('review_profilename').review_overall.count().to_frame('count_reviews').sort_values(by = "count_reviews", ascending = False)
reviews_count_by_profilename.reset_index(inplace=True)
reviews_count_by_profilename.head()
reviews_count_by_profilename.count()
#now lets explore number of users with number of reviews they have given
reviews_count_user = reviews_count_by_profilename.count_reviews.value_counts().to_frame().reset_index()
reviews_count_user.columns = ['number_of_reviews','number_of_users']
reviews_count_user.head()
reviews_count_user.count()
reviews_count_user['percentage_users'] = (reviews_count_user['number_of_users']*100)/reviews_count_user.number_of_users.sum()
reviews_count_user['cumul_percentage_users'] = reviews_count_user.percentage_users.cumsum()
reviews_count_user.head()
#visualizing number of users with number of reviews

sns.set(rc={'figure.figsize':(25,7)})
ax = sns.barplot(x=reviews_count_user[0:100].number_of_reviews, y=reviews_count_user[0:200].number_of_users, palette="rocket")
ax.set(yscale="log")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10)
ax
#unique values for each metric
df.beer_beerid.nunique()
df.review_profilename.nunique()
df.review_overall.nunique()
df.beer_beerid.count()
beer_ids_no_of_ratings_grt_30 = reviews_count_beer_id.loc[reviews_count_beer_id.Reviews_count>=30].beer_beerid.to_frame('beer_beerid')
users_no_of_ratings_grt_30 = reviews_count_by_profilename.loc[reviews_count_by_profilename.count_reviews>=30].review_profilename.to_frame('review_profilename')
rating2 = pd.merge(df, beer_ids_no_of_ratings_grt_30, how='inner', on='beer_beerid')
rating2 = pd.merge(rating2, users_no_of_ratings_grt_30, how='inner', on='review_profilename')
rating2.head()
rating2.count()
np.sort(rating2.review_overall.unique())
rating2.head()
rating2.groupby('beer_beerid').review_overall.mean()
plt.hist(rating2.groupby('beer_beerid').review_overall.mean(), color="m")
#sns.distplot(rating2.groupby('beer_beerid').review_overall.mean(), kde=False, color="b")
sns.distplot(rating2.groupby('beer_beerid').review_overall.mean(), color="m")
rating2.groupby('beer_beerid').review_overall.mean().mean()
rating2.groupby('review_profilename').review_overall.mean()
plt.hist(rating2.groupby('review_profilename').review_overall.mean(), color="m")
rating2.groupby('review_profilename').review_overall.mean().mean()
#we already deleted beerid records having only upto 5 ratings, so lets reform this dataframe.
reviews_count_beer_id2 = rating2.groupby('beer_beerid').review_overall.count().to_frame('Reviews_count').sort_values(by = "Reviews_count", ascending = False)
reviews_count_beer_id2.head()
sns.distplot(reviews_count_beer_id2, color="m")
reviews_count_beer_id2.Reviews_count.mean()


review_count_user = rating2.groupby('review_profilename').review_overall.count().to_frame('Reviews_count').sort_values(by = "Reviews_count", ascending = False)
review_count_user.head()
sns.distplot(review_count_user, color="m")
review_count_user.Reviews_count.mean()
import sklearn
from sklearn.model_selection import train_test_split
train, test = train_test_split(rating2, test_size=0.30, random_state=31)
print(train.shape)
print(test.shape)
train.head()
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)
train.head()
# pivot ratings beer features
df_beer_features = train.pivot_table(index='review_profilename',columns='beer_beerid',values='review_overall').fillna(0)
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

#User Similarity Matrix
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
#Finding the top 5 recommendation for the user 1
user_final_rating.iloc[0].sort_values(ascending=False)[0:5]
beer_features = train.pivot_table( index='review_profilename', columns='beer_beerid', values='review_overall' ).T

beer_features.head()

mean = np.nanmean(beer_features, axis=1)
df_subtracted1 = (beer_features.T-mean).T
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
#Top 5 prediction for the user -1


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
item_correlation.shape
item_correlation1 = item_correlation

np.fill_diagonal(item_correlation1, 0)
b = largest_indices(item_correlation1, 20)
b
item_correlation1[largest_indices(item_correlation1, 20)]
#heat map for comparing ratings of similar beers
#index to beer_id mapping
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
#Using Item Similarity
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
