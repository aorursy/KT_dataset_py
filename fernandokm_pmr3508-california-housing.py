import numpy as np
import pandas as pd
import pandas_profiling

import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, f_regression
from sklearn.pipeline import Pipeline
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import pearsonr
import seaborn as sns

from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from IPython import display
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')

ytrain = train['median_house_value']
train.info()
pandas_profiling.ProfileReport(train)
all_features = train.columns
all_features.drop('median_house_value')

def dict_pearsonr(a):
    corr, pval = pearsonr(a, train['median_house_value'])
    fcorr, fpval = pearsonr(a >= a.mean(), train['median_house_value'])
    return {'correlation [x]': corr, 'p-value [x]': pval}
stats = train[all_features].apply(dict_pearsonr, result_type='expand').transpose()
stats[['correlation [x]']].plot(kind='bar', title='Correlations', figsize=(10, 5))
stats[['p-value [x]']].plot(kind='bar', title='P-values', figsize=(10, 5))
plt.yscale('log')
from sklearn.ensemble import IsolationForest

isol = IsolationForest(behaviour='new', contamination='auto')
results = isol.fit_predict(train) # returns -1 if outlier
inliers, = (results+1).nonzero()
print('Percentage of detected outliers:', 1 - inliers.size/ytrain.size)
min_lon = train['longitude'].min() - 1
max_lon = train['longitude'].max() + 1
min_lat = train['latitude'].min() - 1
max_lat = train['latitude'].max() + 1
width=10

fig = plt.figure(figsize=(width, width*(max_lat-min_lat)/(max_lon-min_lon)))

m = Basemap(min_lon, min_lat, max_lon, max_lat, resolution='i')
m.drawmapboundary()
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmeridians

x, y = m(train['longitude'][inliers], train['latitude'][inliers])
plt.scatter(x, y, c=ytrain[inliers], cmap='YlOrRd', alpha=.2, s=15)
plt.colorbar()
# We will consider only the median_house_value >= 500000 (which, as seen in the profile above, represents the top 5% of the houses)
ap = AffinityPropagation()
ap.fit(train.iloc[inliers][ytrain[inliers] >= 500000][['longitude', 'latitude']])
print('Number of clusters:', ap.cluster_centers_.shape[0])
min_lon = train['longitude'].min() - 1
max_lon = train['longitude'].max() + 1
min_lat = train['latitude'].min() - 1
max_lat = train['latitude'].max() + 1
width=10

fig = plt.figure(figsize=(width, width*(max_lat-min_lat)/(max_lon-min_lon)))

m = Basemap(min_lon, min_lat, max_lon, max_lat, resolution='i')
m.drawmapboundary()
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmeridians

x, y = m(train['longitude'], train['latitude'])
plt.scatter(x, y, c=ytrain, cmap='YlOrRd', alpha=.2, s=15)
plt.colorbar()

xc, yc = m(ap.cluster_centers_[:, 0], ap.cluster_centers_[:, 1])
plt.scatter(xc, yc, c='b', alpha=1, s=20)
def fs1(data):
    data = data.copy().astype('float64')
    data.drop(['Id', 'median_house_value'], axis=1, errors='ignore', inplace=True)
    return data

xtrain1 = fs1(train)
xtest1 = fs1(test)
def fs2(data):
    data = fs1(data)
    data['rpp'] = data['total_rooms'] / data['population']
    data['rph'] = data['total_rooms'] / data['households']
    data['bpp'] = data['total_bedrooms'] / data['population']
    data['bph'] = data['total_bedrooms'] / data['households']
    data['trpp'] = (data['total_rooms'] + data['total_bedrooms']) / data['population']
    data['trph'] = (data['total_rooms'] + data['total_bedrooms']) / data['households']
    data['pph'] = data['population'] / data['households']
    
    for i in range(ap.cluster_centers_.shape[0]):
        x, y = ap.cluster_centers_[i]
        data[f'd{i}'] = np.sqrt((data['longitude'] - x)**2 + (data['latitude'] - y)**2)
    return data

xtrain2 = fs2(train)
xtest2 = fs2(test)
def fs3(data):
    data = fs2(data)
    data.drop(['households', 'total_bedrooms', 'bpp', 'bph', 'trpp', 'trph'], axis=1, inplace=True)
    return data

xtrain3 = fs3(train)
xtest3 = fs3(test)
rmsle = make_scorer(lambda y1, y2: np.sqrt(mean_squared_log_error(y1, y2)), greater_is_better=False)
pipe = Pipeline([
    ('ss', MinMaxScaler()),
    ('lr', LinearRegression()),
])

def count_negative_predictions(predictor, x, y):
    predictor.fit(x, y)
    return sum(predictor.predict(x) < 0)

count_negative_predictions(pipe, xtrain1, ytrain)
count_negative_predictions(pipe, xtrain2, ytrain)
count_negative_predictions(pipe, xtrain3, ytrain)
pipe = Pipeline([
    ('ss', MinMaxScaler()),
    ('lr', Ridge()),
])

count_negative_predictions(pipe, xtrain2, ytrain)
pipe = Pipeline([
    ('ss', MinMaxScaler()),
    ('sfm', SelectFromModel(Lasso(max_iter=1000000, precompute=True))),
    ('lr', LinearRegression())
])

count_negative_predictions(pipe, xtrain2, ytrain)
pipe = Pipeline([
    ('PCA', PCA()),
    ('knn', KNeighborsRegressor()),
])
gs = GridSearchCV(pipe, {'knn__n_neighbors': list(range(1,20))}, scoring=rmsle, cv=3)
gs.fit(xtrain2, ytrain)
gs.best_score_, gs.best_params_
gs = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': list(range(1,8))}, scoring=rmsle, cv=3)
xxtrain = xtrain2.copy()
xxtrain[['longitude', 'latitude']] *= 1e5
gs.fit(xxtrain, ytrain)
gs.best_score_, gs.best_params_
gs = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': list(range(1,8))}, scoring=rmsle, cv=3)
xxtrain = xtrain2.copy()
xxtrain[['longitude', 'latitude']] *= 1e6
gs.fit(xxtrain, ytrain)
gs.best_score_, gs.best_params_
gs = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': list(range(1,8))}, scoring=rmsle, cv=3)
xxtrain = xtrain2.copy()
xxtrain[['longitude', 'latitude']] *= 1e7
gs.fit(xxtrain, ytrain)
gs.best_score_, gs.best_params_
rf = RandomForestRegressor(n_estimators=10, max_depth=1)
cross_val_score(rf, xtrain2, ytrain, scoring=rmsle, cv=3).mean()
gs = GridSearchCV(rf, {'n_estimators': [5, 10, 15], 'max_depth': [1,5,10]}, scoring=rmsle, cv=3)
gs.fit(xtrain2, ytrain)
gs.best_score_, gs.best_params_
ab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=10)
cross_val_score(ab, xtrain2, ytrain, scoring=rmsle, cv=3).mean()
gs = GridSearchCV(ab, {'base_estimator__max_depth': [1, 2, 5, 10], 'n_estimators': [5,10,15]}, scoring=rmsle, cv=3)
gs.fit(xtrain2, ytrain)
gs.best_score_, gs.best_params_
def generate_submission(classifier, xtrain, xtest, out_file):
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    df = pd.DataFrame({'Id': test['Id'], 'median_house_value': ypred})
    df.to_csv(out_file, index=False)
    
    score = cross_val_score(classifier, xtrain, ytrain, scoring=rmsle, cv=5).mean()
    print('{0}: score={1:.3f}\n'.format(out_file, score))
# Escolhendo os melhores resultados obtidos acima

xxtrain = xtrain2.copy()
xxtrain[['longitude', 'latitude']] *= 1e5
xxtest = xtest2.copy()
xxtest[['longitude', 'latitude']] *= 1e5
generate_submission(KNeighborsRegressor(n_neighbors=7), xxtrain, xxtest, 'sub1.csv')

xxtrain = xtrain2.copy()
xxtrain[['longitude', 'latitude']] *= 1e6
xxtest = xtest2.copy()
xxtest[['longitude', 'latitude']] *= 1e6
generate_submission(KNeighborsRegressor(n_neighbors=4), xxtrain, xxtest, 'sub2.csv')

generate_submission(AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=10), xtrain2, xtest2, 'sub3.csv')

generate_submission(AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=15), xtrain2, xtest2, 'sub4.csv')