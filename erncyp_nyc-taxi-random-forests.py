# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
train_df =  pd.read_csv('../input/train.csv', nrows = 5_000_000)
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
train_df['distance'] = haversine_np(train_df['pickup_longitude'], train_df['pickup_latitude'], 
                                    train_df['dropoff_longitude'], train_df['dropoff_latitude'])
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime']) 
train_df['year'] = train_df['pickup_datetime'].dt.year
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['day'] = train_df['pickup_datetime'].dt.day
train_df['hour'] = train_df['pickup_datetime'].dt.hour
train_df['minute'] = train_df['pickup_datetime'].dt.minute
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))
new_york_lat = 40
new_york_long = -74
train_df.describe()
cond = True
for col in {'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'}:
    cond &= abs(train_df[col] - train_df[col].mean()) < 5

print('Old size: %d' % len(train_df))
train_df = train_df[cond]
print('New size: %d' % len(train_df))
train_df.describe()
for col in {'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'}:
    train_df['rough' + col] = train_df[col].round(2)
a=train_df.groupby(['roughpickup_latitude','roughpickup_longitude'])[['pickup_latitude', 'pickup_longitude']].agg(['mean','count'])
a.columns = ['mean_pickup_latitude','c1','mean_pickup_longitude','c2']
a.head()
a.sort_values(['c1','c2'],ascending=False).head()
# this is the pensylvania station
b=train_df.groupby(['roughdropoff_latitude','roughdropoff_longitude'])[['dropoff_latitude', 'dropoff_longitude']].agg(['mean','count'])
b.columns = ['mean_dropoff_latitude','c1','mean_dropoff_longitude','c2']
b.head()
b.sort_values(['c1','c2'],ascending=False).head(n=20)
# this is near a hotel also near the station?
# what about the airports?
# jfk 40.6413151,-73.7803278,
b[ b['mean_dropoff_latitude']<40.7].sort_values(['c1','c2'],ascending=False).head()

b['c1'].plot.hist()
plt.title('occurance of counts of rough dropoff locations')
plt.yscale('log')
# this figure is showing us that while most locations have hardly any unique dropoffs, quitea few places are getting
# loads of counts
a['c1'].plot.hist()
plt.title('occurance of counts of rough dropoff locations')
plt.yscale('log')
a[a['c1']>1000].shape
a.shape
a = a[['c1']].reset_index().rename(columns={'c1':'pickup_busyness'})
b = b[['c1']].reset_index().rename(columns={'c1':'dropoff_busyness'})
train_df = pd.merge(train_df,a, how='left')
train_df = pd.merge(train_df,b, how='left')
train_df.head()
X = train_df[['distance','year','month','day','hour','pickup_busyness','dropoff_busyness']].values
Y = train_df['fare_amount'].values
from sklearn.ensemble import RandomForestRegressor
# from hyper parameter tuning:
kwargs = {'bootstrap': True,
 'max_depth': None,
 'max_features': 3,
 'min_samples_leaf': 9,
 'min_samples_split': 2}
rand_regr = RandomForestRegressor(n_estimators=20, **kwargs)
rand_regr.fit(X, Y)
y_pred = rand_regr.predict(X)
print('chi squared  rand forest with date %s' % (np.sum((Y-y_pred)**2.)/len(Y))**0.5)
rand_regr.score(X,Y)
# from sklearn.model_selection import RandomizedSearchCV
# from time import time
# from scipy.stats import randint as sp_randint
# build a classifier
# clf = RandomForestRegressor(n_estimators=20)
# Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")


# # specify parameters and distributions to sample from
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 6),
#               "min_samples_split": sp_randint(2, 11),
#               "min_samples_leaf": sp_randint(1, 11),
#               "bootstrap": [True, False],
# #               "criterion": ["gini", "entropy"]
#              }

# # run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=3)

# start = time()
# random_search.fit(X, Y)
# # print("RandomizedSearchCV took %.2f seconds for %d candidates"
# #       " parameter settings." % ((time() - start), n_iter_search))
# # report(random_search.cv_results_)
# pd.DataFrame(random_search.cv_results_)
# random_search.best_estimator_
# random_search.best_params_
# random_search.best_score_
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(), X, Y)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1,
#                  color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation score")


# plt.legend(loc="best")
test_df =  pd.read_csv('../input/test.csv')
test_df['distance'] = haversine_np(test_df['pickup_longitude'], test_df['pickup_latitude'], 
                                    test_df['dropoff_longitude'], test_df['dropoff_latitude'])
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime']) 
test_df['year'] = test_df['pickup_datetime'].dt.year
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['minute'] = test_df['pickup_datetime'].dt.minute
for col in {'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'}:
    test_df['rough' + col] = test_df[col].round(2)
test_df = pd.merge(test_df,a, how='left')
test_df = pd.merge(test_df,b, how='left')
test_df['pickup_busyness'] = test_df['pickup_busyness'].fillna(1)
test_df['dropoff_busyness'] = test_df['dropoff_busyness'].fillna(1)
X_to_pred = test_df[['distance','year','month','day','hour','pickup_busyness','dropoff_busyness']].values
y_pred = rand_regr.predict(X_to_pred)
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

# x=np.arange(0,500)
# y=regr.predict(x.reshape(-1,1))
# plt.plot(X,Y,'o')
# plt.plot(x,y)
# plt.show()
# with sns.axes_style("white"):
#     sns.jointplot(x=X.flatten(), y=Y, kind="hex", color="k", bins='log');
# X = X.flatten()
# mask = (X <50) & (Y <100)
# # mask
# # fig, ax = plt.subplots()

# with sns.axes_style("white"):
#     p = sns.jointplot(x=X[mask], y=Y[mask], kind="hex", color="k", bins='log');

# x=np.arange(0,50)
# y=regr.predict(x.reshape(-1,1))
# p.ax_joint.plot(x,y)
# from sklearn.cluster import KMeans
# X_clus = train_df[['fare_amount','distance']].values
# km = KMeans(n_clusters=5,random_state=0)
# km.fit(X_clus)
# plt.scatter(X_clus[:,0],X_clus[:,1], c=km.predict(X_clus))
# plt.show()
# from sklearn.cluster import MiniBatchKMeans
# kmm = MiniBatchKMeans(n_clusters=5, random_state=0)
# kmm.fit(X_clus)
# plt.scatter(X_clus[:,0],X_clus[:,1], c=kmm.predict(X_clus))
# plt.show()
# plt.hist(X_clus[:,0]/(X_clus[:,1]+1e-10),bins=100,range=[0,60])
# plt.show()

# from sklearn.cluster import DBSCAN
# dbs = DBSCAN(eps=0.3, min_samples=10)
# dbs.fit(X_clus)
# plt.scatter(X_clus[:,0],X_clus[:,1], c=dbs.predict(X_clus))
# plt.show()

