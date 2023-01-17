import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import os
print(os.listdir())
train = pd.read_csv('../input/train.csv', index_col=0)
train.head()
train.describe()
test = pd.read_csv('../input/test.csv', index_col=0)
test.head()
x_train = train.drop(['median_house_value'], axis=1)
x_train.head()
y_train = train['median_house_value']
y_train.head()
new_test = test.drop(['latitude', 'longitude'], axis=1)
for i in range(1, 9):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=train.iloc[:,i-1], y=train['median_house_value'], s=3)
    ax.set(xlabel=train.columns[i-1], ylabel='median_house_value')
    plt.show()
for i in range(2, 9):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=np.log(train.iloc[:,i-1]), y=np.log(train['median_house_value']), s=3)
    ax.set(xlabel=train.columns[i-1], ylabel='median_house_value')
    plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure() #.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train['longitude'], train['latitude'], train['median_house_value'], s=3)
plt.show()
new_train = train.assign(
    mean_rooms=lambda df: df['total_rooms']/df['households']).assign(
    mean_bedrooms=lambda df: df['total_bedrooms']/df['households'])

new_train.head()
for i in range(10, 12):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=np.log(new_train.iloc[:,i-1]), y=np.log(train['median_house_value']), s=3)
    ax.set(xlabel=new_train.columns[i-1], ylabel='median_house_value')
    plt.show()
x_train_new = new_train.drop(['latitude', 'longitude', 'total_rooms', 'total_bedrooms', 'median_house_value'], axis=1)
x_train_new.head()
new_test = test.assign(
    mean_rooms=lambda df: df['total_rooms']/df['households']).assign(
    mean_bedrooms=lambda df: df['total_bedrooms']/df['households']).drop(
    ['latitude', 'longitude', 'total_rooms', 'total_bedrooms'], axis=1)
new_test.head()
import geopy as gp
richest_cities = ['Santa Clara', 'Marin', 'San Mateo', 'Contra Costa', 'San Francisco', 'Atherton', 'Woodside', 
                 'Hidden Hills', 'Los Altos Hills', 'Belvedere, CA', 'Beverly Hills', 'Santa Monica', 'Los Angeles',
                 'San Diego', 'Sacramento']
poorest_cities = ['Mendota CA', 'Adelanto CA', 'Clearlake CA', 'Orange Cove', 'Corning CA', 'Arcata', 'Mcfarland',
                 'Parlier', 'Calipatria', 'Woodlake']
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='competition')
rich_coord = [(geolocator.geocode(city).latitude, geolocator.geocode(city).longitude) for city in richest_cities]
for city, c in zip(richest_cities, rich_coord):
    print('City: {}, Coord: {}'.format(city,c))
poor_coord = [(geolocator.geocode(city).latitude, geolocator.geocode(city).longitude) for city in poorest_cities]
for city, c in zip(poorest_cities, poor_coord):
    print('City: {}, Coord: {}'.format(city,c))
coord = rich_coord + poor_coord
from geopy.distance import distance
from copy import deepcopy
npcoord = np.array(coord)
x_train3 = deepcopy(x_train)
test3 = deepcopy(test)
#x_train3 = x_train.assign(dist1=lambda df:np.sqrt((df['latitude']-coord[0][0])**2+(df['longitude']-coord[0][1])**2))
for i, c in enumerate(npcoord):
    x_train3['dist{}'.format(i+1)] = np.sqrt((x_train['latitude']-c[0])**2+(x_train['longitude']-c[1])**2)
    test3['dist{}'.format(i+1)] = np.sqrt((test['latitude']-c[0])**2+(test['longitude']-c[1])**2)
x_train3.head()
for i in range(1, 26):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=x_train3.loc[:,'dist{}'.format(i)], y=train['median_house_value'], s=3)
    ax.set(xlabel='dist{}'.format(i), ylabel='median_house_value')
    plt.show()
for i in range(1, 26):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=np.log(x_train3.loc[:,'dist{}'.format(i)]), y=np.log(train['median_house_value']), s=3)
    ax.set(xlabel='dist{}'.format(i), ylabel='median_house_value')
    plt.show()
x_train4 = deepcopy(x_train)
test4 = deepcopy(test)
for i in range(1, 26):
    x_train4['log_dist{}'.format(i)] = np.log(x_train3['dist{}'.format(i)])
    test4['log_dist{}'.format(i)] = np.log(test3['dist{}'.format(i)])
x_train4.head()
x_train5 = x_train4.drop(['latitude', 'longitude', 'total_rooms', 'total_bedrooms'], axis=1)
test5 = test4.drop(['latitude', 'longitude', 'total_rooms', 'total_bedrooms'], axis=1)
x_train5.head()
x_train6 = x_train5.drop(['population', 'median_age', 'households'], axis=1)
test6 = test5.drop(['population', 'median_age', 'households'], axis=1)
x_train6.head()
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=7).fit(x_train5, y_train)
best_indices = selector.get_support(indices=True)
x_train7 = x_train5.iloc[:,best_indices]
test7 = test5.iloc[:,best_indices]
x_train7.head()
x_train8 = deepcopy(x_train4)
test8 = deepcopy(test4)
x_train8['people_per_house'] = x_train8['population']/x_train8['households']
test8['people_per_house'] = test8['population']/test8['households']
x_train8['rooms_per_house'] = x_train8['total_rooms']/x_train8['households']
test8['rooms_per_house'] = test8['total_rooms']/test8['households']
x_train8['bedrooms_per_house'] = x_train8['total_bedrooms']/x_train8['households']
test8['bedrooms_per_house'] = test8['total_bedrooms']/test8['households']
x_train8.head()

for i in range(24, 27):
    fig, ax = plt.subplots()
    #plt.subplot(4,2,i)
    plt.scatter(x=np.log(x_train8.iloc[:,i-1]), y=np.log(train['median_house_value']), s=3)
    ax.set(xlabel=x_train8.columns[i-1], ylabel='median_house_value')
    plt.show()
x_train9 = deepcopy(x_train8)
test9 = deepcopy(test8)

x_train9.drop(['latitude', 'longitude', 'total_rooms', 'total_bedrooms'], axis=1, inplace=True)
test9.drop(['latitude', 'longitude', 'total_rooms', 'total_bedrooms'], axis=1, inplace=True)
x_train9.head()
def create_csv(pred, test, name):
    results = np.vstack((test.index, pred)).T
    cols = ['Id', 'median_house_value']
    df_pred = pd.DataFrame(columns=cols ,data=results)
    df_pred['Id'] = df_pred['Id'].astype('Int32') 
    df_pred.to_csv(name, index=False)
from sklearn.metrics.scorer import make_scorer
def rmsle_func(y_test, y_pred, **kwargs): 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

rmsle = make_scorer(rmsle_func, greater_is_better=True)
from sklearn.model_selection import cross_val_score
def get_cv(model, x, y, cv=10):
    score = cross_val_score(model, x, y, cv=cv, scoring=rmsle)
    return score.mean()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lin_reg = LinearRegression()
lin_score = cross_val_score(lin_reg, x_train_new, y_train, cv=10)
lin_score.mean()
new_lin_score = get_cv(lin_reg, x_train_new, y_train)
new_lin_score
lin_score_1f = get_cv(lin_reg, x_train[['median_income']], y_train)
lin_score_1f
lin_reg.fit(x_train_new, y_train)
lin_pred = lin_reg.predict(new_test)
positive_lin_pred = [abs(x) for x in lin_pred]
positive_lin_pred
create_csv(positive_lin_pred, test, 'new_lin_reg_pred.csv')
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_score = cross_val_score(ridge_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
ridge_score.mean()
ridge_reg.fit(x_train, y_train)
ridge_pred = ridge_reg.predict(test)
new_ridge_pred = [x if x > 0 else 0 for x in ridge_pred]
create_csv(new_ridge_pred, test, 'ridge_pred.csv')
from sklearn.ensemble import RandomForestRegressor
rforest = RandomForestRegressor(n_estimators=250, random_state=0)
rfor_scores = {}
for i in range(1, 151, 15):
    rforest = RandomForestRegressor(n_estimators=i, random_state=0)
    score = get_cv(rforest, x_train_new, y_train)
    rfor_scores[i] = score
    print('{}: {};'.format(i, score), end=' ')
plt.scatter(rfor_scores.keys(), rfor_scores.values())
plt.show()
rfor_score = get_cv(rforest, x_train9, y_train)
rfor_score
rfor_score_1f = get_cv(rforest, x_train[['median_income']], y_train)
rfor_score_1f
rforest.fit(x_train4, y_train)
rfor_pred = rforest.predict(test4)
rfor_pred
create_csv(rfor_pred, test, 'random_forest_pred12.csv')
import xgboost as xgb
from xgboost import XGBRegressor
xgscores = {}
xgscores['complete'] = {}
xgscores['analysed'] = {}
xgscores['1f'] = {}
for depth in range(1, 4):
    model = XGBRegressor(max_depth=depth)
    xgscores['complete'][depth] = get_cv(model, x_train, y_train)
    xgscores['analysed'][depth] = get_cv(model, x_train_new, y_train)
    xgscores['1f'][depth] = get_cv(model, x_train[['median_income']], y_train)
xgscores
xgscores_complete = {}
for depth in range(8, 13):
    model = XGBRegressor(max_depth=depth)
    xgscores_complete[depth] = get_cv(model, x_train4, y_train)
    print(f'{depth}: {xgscores_complete[depth]}')
xgscores_complete
xgreg = XGBRegressor(max_depth=10) # 0.22337
xgreg.fit(x_train9, y_train)
xgpred = xgreg.predict(test9)
xgpred
create_csv(xgpred, test, 'xgboost_pred13.csv')
