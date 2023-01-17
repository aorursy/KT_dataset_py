# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine Learning
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb
# Read data
train = pd.read_csv('../input/train.csv', nrows = 1000000)
test = pd.read_csv('../input/test.csv')
# Let's start by checking for NaN values
print('Sum of NaN values for each column')
print(train.isnull().sum())

# It seems like we lost some data for the dropoff. There are several ways of handling this, but I just go with removing the rows.
train = train.dropna()
print('Sum of NaN values for each column after dropping NaN')
print(train.isnull().sum())
# Seems like we have a few outliers. Let's visualize the data and see if we can spot the outliers.
#columns_to_plot = ['fare_amount', 'passenger_count']
#sns.pairplot(train.loc[:, train.columns != 'pickup_datetime'])
pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_latitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()

# So there seem to be a lot of outliers:

# Latitude and longitude varies from -3116.28 to 2522.27 whereas the mean is around 40 (pickup_latitude, but goes for all the coordinates)
# This is probably due to a typo when data was gathered. Let's select a more reasonable value (2 times the standard deviation)
#columns_to_select = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
#for column in columns_to_select:
#    train = train.loc[(train[column] > train[column].mean() - train[column].std() * 2) & (train[column] < train[column].mean() + train[column].std() * 2)]

# Manually picking reasonable levels
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 300)]
train = train.loc[(train['pickup_longitude'] > pickup_longitude_min) & (train['pickup_longitude'] < pickup_longitude_max)]
train = train.loc[(train['pickup_latitude'] > pickup_latitude_min) & (train['pickup_latitude'] < pickup_latitude_max)]
train = train.loc[(train['dropoff_longitude'] > dropoff_longitude_min) & (train['dropoff_longitude'] < dropoff_longitude_max)]
train = train.loc[(train['dropoff_latitude'] > dropoff_latitude_min) & (train['dropoff_latitude'] < dropoff_latitude_max)]
# Let's assume taxa's can be mini-busses as well, so we select up to 8 passengers.
train = train.loc[train['passenger_count'] <= 8]
train.describe()
def haversine_np(a):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = a
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def cityblock(a,dist='cityblock'):
    return pdist(a.reshape(2,2),dist)[0]

# from multiprocessing import Pool
# pool = Pool()
# coords = train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].values
# a = pool.map(haversine_np,coords)
# del pool
dist_types = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
dist_types = ['chebyshev','euclidean','canberra','sqeuclidean','braycurtis','minkowski','hamming','cityblock']
from multiprocessing import Pool
pool = Pool()
combine = [train, test]
for dataset in combine:    
    coords = dataset[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].values

    dataset['haversine'] = pool.map(haversine_np,coords)
        
    
    for dist_type in dist_types:
        print(dist_type)
        dataset[dist_type] = pool.starmap(cityblock,[(x,dist_type,) for x in coords],1000)
    
    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival
    # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
pool.close()
pool.join()
del pool
train.head(3)
features = [ 'pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
             'haversine', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis',
             'minkowski',  'hamming', 'cityblock']

n = StandardScaler()
train[features] = n.fit_transform(train[features])
test[features] = n.transform(test[features])
coords = ['pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',]
concat = pd.concat([train[coords],test[coords]])
db = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True).fit(concat)
labels = db.labels_
train['cluster'] = labels[:train.shape[0]]
test['cluster'] = labels[train.shape[0]:]

db = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True).fit(concat[['pickup_latitude',  'pickup_longitude']])
labels = db.labels_
train['cluster1'] = labels[:train.shape[0]]
test['cluster1'] = labels[train.shape[0]:]

db = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True).fit(concat[['dropoff_latitude', 'dropoff_longitude',]])
labels = db.labels_
train['cluster2'] = labels[:train.shape[0]]
test['cluster2'] = labels[train.shape[0]:]
pca_features = [ 
#     'pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
             'haversine', 'chebyshev', 'euclidean', 'canberra',  'braycurtis',
             'minkowski',  'hamming', 'cityblock']

pca = PCA(n_components=3)
p_result = pca.fit_transform(train[pca_features])
for x in range(p_result.shape[1]):
    train['pca0' + str(x)] = p_result[:,x]
  
p_result = pca.transform(test[pca_features])
for x in range(p_result.shape[1]):
    test['pca0' + str(x)] = p_result[:,x]

# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
good_dist =  ['pca0','pca1','pca2','cluster',
#               'haversine','chebyshev','euclidean','canberra','braycurtis','minkowski','hamming','cityblock'
             ]
# Let's drop all the irrelevant features
train_features_to_keep = ['fare_amount',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist
train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)

test_features_to_keep = ['key',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist
test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)

# Let's prepare the test set
x_pred = test.drop('key', axis=1)
# Let's run XGBoost and predict those fares!
x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train['fare_amount'],random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse',
                            'eta':.3,
                            'max_depth':4,
                            'min_child_weight':3,
                           }
                    ,dtrain=matrix_train,num_boost_round=300,
                    early_stopping_rounds=10,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb.plot_importance(model)
plt.show()
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)
submission
