import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
from sklearn.decomposition import PCA
desti1 = pd.read_csv("../input/destinations.csv")
desti1.info()
X = desti1.ix[:,1:150]
X.info()
y = desti1.ix[:,0:1]
y.info()
pca = PCA(n_components=5)
X_r = pca.fit(X).transform(X)
X_r1 = pd.DataFrame(X_r)
X_r1["srch_destination_id"] = desti1["srch_destination_id"]
print('explained variance ratio (first 5 components): %s' % str(pca.explained_variance_ratio_))
X_r1.info()

X_r1.columns = ['d1','d2','d3','d4','d5','srch_destination_id']
train1 = pd.read_csv("../input/train.csv", nrows=1000000)
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
test1["srch_co"] = pd.to_datetime(test1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[D]')
test1["stay_span"] = (test1["srch_co"] - test1["srch_ci"]).astype('timedelta64[D]')
train2 = pd.to_datetime(train1["date_time"])
test2 = pd.to_datetime(test1["date_time"])
train2 = pd.DataFrame(train2)
test2 = pd.DataFrame(test2)
test1['year'] = test2['date_time'].dt.year
test1['month'] = test2['date_time'].dt.month
test1['day_of_week'] = test2['date_time'].dt.dayofweek
test1['day'] = test2['date_time'].dt.day
test1['hour'] = test2['date_time'].dt.hour
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
test1["date_time"] = pd.to_datetime(test1["date_time"], format='%Y-%m-%d', errors="coerce")
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["search_span"] = (test1["srch_ci"] - test1["date_time"]).astype('timedelta64[D]')
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
test1.ix[(test1['hour'] >= 10) & (test1['hour'] < 18), 'hour'] = 1
test1.ix[(test1['hour'] >= 18) & (test1['hour'] < 22), 'hour'] = 2
test1.ix[(test1['hour'] >= 22) & (test1['hour'] == 24), 'hour'] = 3
test1.ix[(test1['hour'] >= 1) & (test1['hour'] < 10), 'hour'] = 3
train1 = train1.drop('srch_ci', axis=1)
test1 = test1.drop('srch_ci', axis=1)
train1 = train1.drop('srch_co', axis=1)
test1 = test1.drop('srch_co', axis=1)
train1 = train1.drop('date_time', axis=1)
test1 = test1.drop('date_time', axis=1)
train1 = train1.join(X_r1, on = 'srch_destination_id', how = 'left', rsuffix='dest')
train1 = train1.drop("srch_destination_iddest", axis=1)
train1.fillna(-1, inplace=True)
test1.fillna(-1, inplace=True)
train1.info()
hotelCluster = train1.ix[:,'hotel_cluster']
hotelCluster1 = pd.DataFrame(hotelCluster)
hotelCluster1.info()
hotelCluster1['hotel_cluster'].head()
train1 = train1.drop('hotel_cluster', axis=1) #df.drop('reports', axis=1)
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
clf = MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(100, 3), learning_rate='adaptive', random_state=1)
print(hotelCluster.shape)
clf.fit(train1, hotelCluster)
test1 = pd.read_csv("../input/test.csv", parse_dates=['date_time'], nrows=10)
test1.info()
test1['j'] = -1