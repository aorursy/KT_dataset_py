# Importing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mpl_toolkits as mpl_toolkits
from math import radians, cos, sin, asin, sqrt, log
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#for decision tree
from sklearn.tree import DecisionTreeRegressor
#for clustering\graphics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Dataset reading
frame = pd.read_csv('../input/kc_house_data.csv')
frame.info()
current_year = datetime.now().year
for i, x in frame[['yr_built']].itertuples():
    if(frame.at[i, 'yr_renovated'] == 0):
        frame.at[i, 'yr_renovated'] = x
    frame.at[i, 'age'] = current_year - frame.at[i, 'yr_renovated']
def haversine(lon2, lat2):
    # Seattle center coordinates
    centerLon = -122.352033
    centerLat = 47.622451

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [centerLon, centerLat, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers, 3956 for miles
    return c * r
for i, lat, long in frame[['lat','long']].itertuples():
    frame.at[i,'distance'] = haversine(long, lat)
for i, f, a, b, c, d in frame[['floors','bedrooms','bathrooms','sqft_above','sqft_basement']].itertuples():
    frame.at[i,'average_sqft_by_room'] = ((c+d)/((a+b)+1))#+1 because house can be without bedrooms or bathrooms
    frame.at[i,'average_sqft_by_floor'] = ((c+d)/f)
for i, f, sqft in frame[['floors','sqft_basement']].itertuples():
    frame.at[i,'sqft_floors_mult_basement'] = f*sqft
frame['date'] = pd.to_datetime(frame['date'])
for i, d in frame[['date']].itertuples():
    frame.at[i, 'mnths'] = (d.year - 2014)*12 + d.month

#Delete useless field
del frame['date']
#Convert difference between sqft_living15 and sqft_living to ren_living_diff - sqft difference after renovation 
for i, l, lo, l15, lo15 in frame[['sqft_living','sqft_lot','sqft_living15','sqft_lot15']].itertuples():
    frame.at[i,'ren_living_diff'] = l15-l
    frame.at[i,'ren_lot_diff'] = lo15-lo
# Simple points visualization by x and y coordinates
def draw(x, y):
    data = frame[[x,y]].values
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=.4)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    return data
# Distance by price visualization
dist_price = draw('price','distance')
# Train and centers initializing
k_means_district = KMeans(n_clusters=7)
k_means_district.fit(dist_price)
centers = k_means_district.cluster_centers_
# Districts clustering by price visualization
plt.xlabel("price")
plt.ylabel("distance")
plt.scatter(dist_price[:, 0], dist_price[:, 1], c=k_means_district.predict(dist_price), s=10, alpha=1)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=30, alpha=1)
# Splitting lots into districts
distance_clusters = k_means_district.predict(dist_price)
for i, d in frame[['distance']].itertuples():
    frame.at[i,'price_district'] = distance_clusters[i]
    
pd.DataFrame({
    'price': frame['price'],
    'price_district': frame['price_district']
    }).groupby('price_district').sum().plot()
# Coordinates visualization
coordinates = draw('long','lat')
# Train and centers initialization 
k_means_district.fit(coordinates)
centers = k_means_district.cluster_centers_
plt.xlabel("long")
plt.ylabel("lat")
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=k_means_district.predict(coordinates), s=10, alpha=1)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=10, alpha=1)
# Splitting lots into districts
loc_clusters = k_means_district.predict(coordinates)
for i, lat, long in frame[['lat','long']].itertuples():
    frame.at[i,'loc_district'] = loc_clusters[i]
pd.DataFrame({
    'price': frame['price'],
    'loc_district': frame['loc_district']
    }).groupby('loc_district').sum().plot()
# Distance by price visualization
dist_price = draw('price','distance')
# Train and centers initializing
k_means_neighborhoods = KMeans(n_clusters=127)
k_means_neighborhoods.fit(dist_price)
centers = k_means_neighborhoods.cluster_centers_
# Neigborhoods clustering by price visualization
plt.xlabel("price")
plt.ylabel("distance")
plt.scatter(dist_price[:, 0], dist_price[:, 1], c=k_means_neighborhoods.predict(dist_price), s=10, alpha=1)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=10, alpha=1)
# Splitting lots into neigborhoods
distance_clusters = k_means_neighborhoods.predict(dist_price)
for i, d in frame[['distance']].itertuples():
    frame.at[i,'price_neigborhood'] = distance_clusters[i]

#Visualisation
pd.DataFrame({
    'price': frame['price'],
    'price_neigborhood': frame['price_neigborhood']
    }).groupby('price_neigborhood').sum().plot()
# Train and centers initialization
k_means_neighborhoods.fit(coordinates)
centers = k_means_neighborhoods.cluster_centers_
plt.xlabel("long")
plt.ylabel("lat")
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=k_means_neighborhoods.predict(coordinates), s=10, alpha=1)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=10, alpha=1)
# Splitting lots into neighborhoods
loc_clusters = k_means_neighborhoods.predict(coordinates)
for i, lat, long in frame[['lat','long']].itertuples():
    frame.at[i,'loc_district'] = loc_clusters[i]
pd.DataFrame({
    'price': frame['price'],
    'loc_district': frame['loc_district']
    }).groupby('loc_district').sum().plot()
# Create linear regression object
regr = linear_model.LinearRegression()
def split(frame, features_columns, target_column):
    # Split the frame into training/testing sets
    train_frame, test_frame = train_test_split(frame, test_size = 0.2, random_state=49)
    
    train_features = train_frame.iloc[:,features_columns]
    test_features = test_frame.iloc[:,features_columns]
    
    train_target = train_frame[target_column]
    test_target = test_frame[target_column]
    return train_features, train_target, test_features, test_target
def fit_and_test(train_features, train_target, test_features, test_target):
    regr.fit(train_features, train_target)
    train_prediction = regr.predict(train_features)
    test_prediction = regr.predict(test_features)
    
    print(":::Train:::")
    mse = mean_squared_error(train_target, train_prediction)
    r2s = r2_score(train_target, train_prediction)
    print("Mean squared error: %.2f" % mse)
    print('Variance score: %.2f' % r2s)
    
    mse = mean_squared_error(test_target, test_prediction)
    r2s = r2_score(test_target, test_prediction)
    print(":::Test:::")
    print("Mean squared error: %.2f" % mse)
    print('Variance score: %.2f' % r2s)
#Fit and visualisation model by single feature
def fit_and_test_visualisation(frm, feature_column, target):
    train_features, train_target, test_features, test_target = split(frame, feature_column, target)
    
    regr.fit(train_features, train_target)
    train_prediction = regr.predict(train_features)
    test_prediction = regr.predict(test_features)
    
    print('Feature # {} : {}'.format(feature_column[0], frm.columns[feature_column]))
    
    print(":::Train:::")
    mse = mean_squared_error(train_target, train_prediction)
    r2s = r2_score(train_target, train_prediction)
    print("Mean squared error: %.2f" % mse)
    print('Variance score: %.2f' % r2s)
    
    mse = mean_squared_error(test_target, test_prediction)
    r2s = r2_score(test_target, test_prediction)
    print(":::Test:::")
    print("Mean squared error: %.2f" % mse)
    print('Variance score: %.2f' % r2s)
    
    
    plt.scatter(test_features, test_target,  color='black')
    plt.plot(test_features, test_prediction, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
for x in [*range(frame.columns.size)]:
    fit_and_test_visualisation(frame, [x], 'price')
# Useless fields removing
if 'id' in frame:
    del frame['id']
if 'zipcode' in frame:
    del frame['zipcode']
if 'sqft_lot' in frame:
    del frame['sqft_lot']
if 'yr_renovated' in frame:
    del frame['yr_renovated']
if 'sqft_living15' in frame:
    del frame['sqft_living15']
if 'sqft_lot15' in frame:
    del frame['sqft_lot15']
if 'ren_lot_diff' in frame:
    del frame['ren_lot_diff']
#visualize rest of features
for x in [*range(frame.columns.size)]:
    fit_and_test_visualisation(frame, [x], 'price')
price_index = 0
features = [*range(frame.columns.size)]
target = features[price_index]
features.remove(target)
train_features, train_target, test_features, test_target = split(frame, features, 'price')
fit_and_test(train_features, train_target, test_features, test_target)
# Split data
train_features, train_target, test_features, test_target = split(frame, features, 'price')
# Initialize regression models
decisionTrees = []
for i in range(1, 10):
    decisionTrees.append(DecisionTreeRegressor(max_depth=i))
    
for i in range(1, 8):
    decisionTrees.append(DecisionTreeRegressor(max_depth=i*10))
len(decisionTrees)
def fit_and_visualize(trees):
    depths = []
    # fit regressions models
    for t in trees:
        t.fit(train_features, train_target)
    # Predict
    predictions = []
    for t in trees:
        predictions.append(t.predict(test_features))
    train_scores = []
    test_scores = []
    depths = []
    test_the_best_depths = 0
    max_test = 0
    train_the_best_depths = 0
    max_train = 0
    for t in trees:
        train_score = t.score(train_features,train_target)
        test_score = t.score(test_features,test_target)
        depth = t.max_depth
        depths.append(depth)
        train_scores.append(train_score)
        test_scores.append(test_score)
        if(train_score > max_train):
            max_train = train_score
            train_the_best_depths = depth
        if(test_score > max_test):
            max_test = train_score
            test_the_best_depths = depth
        print("{} level tree train score: {}".format(depth,train_score))
        print("{} level tree test score: {}\n".format(depth,test_score))
    # Visualize
    pd.DataFrame({
    'tree_depth': depths,
    'train score': train_scores,
    'test score': test_scores
    }).groupby('tree_depth').sum().plot()
    
    return train_the_best_depths, test_the_best_depths
#reinitialize
smallDecisionTrees = []
for i in range(1, 100):
    smallDecisionTrees.append(DecisionTreeRegressor(max_depth=i))
train_the_best_depths, test_the_best_depths = fit_and_visualize(smallDecisionTrees)
print("The best train depth: {}".format(train_the_best_depths))
print("The best test depth: {}".format(test_the_best_depths))