import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
destinations = pd.read_csv("../input/destinations.csv")
train1 = pd.read_csv("../input/train.csv", nrows=100000)
test1 = pd.read_csv("../input/test.csv", nrows = 100)
train2 = pd.read_csv("../input/train.csv", nrows=100000)
train3 = pd.read_csv("../input/train.csv", nrows=200002)
train4.info()
train4 = train3.ix[100001:200002,:]
train1["date_time"] = pd.to_datetime(train1["date_time"])
train1["year"] = train1["date_time"].dt.year
train1["month"] = train1["date_time"].dt.month
train1 = train1[train1.is_booking == True]
train4 = train4[train4.is_booking == True]
train4["date_time"] = pd.to_datetime(train4["date_time"])
train4["year"] = train4["date_time"].dt.year
train4["month"] = train4["date_time"].dt.month
from sklearn.decomposition import PCA
X = destinations.ix[:,1:150]
y = destinations.ix[:,0:1]
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)
X_r1 = pd.DataFrame(X_r)
X_r1["srch_destination_id"] = destinations["srch_destination_id"]
print('explained variance ratio (first 3 components): %s' % str(pca.explained_variance_ratio_))
def calc_fast_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    ret = pd.DataFrame(props)
    
    ret = ret.join(X_r1, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret
df = calc_fast_features(train2)
df.fillna(-1, inplace=True)
predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from itertools import chain
all_probs = []
unique_clusters = df["hotel_cluster"].unique()
prediction_frame = pd.DataFrame(all_probs).T
prediction_frame.columns = unique_clusters
import ml_metrics as metrics
def find_top_5(row):
    return list(row.nlargest(5).index)
preds = [ ]
for index, row in prediction_frame.iterrows():
    preds.append(find_top_5(row))
metrics.mapk([[l] for l in train4["hotel_cluster"]], preds, k=5)
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["srch_co"] = pd.to_datetime(test1["srch_co"], format='%Y-%m-%d', errors="coerce")
test1["stay_span"] = (test1["srch_co"] - test1["srch_ci"]).astype('timedelta64[D]')
test2 = pd.to_datetime(test1["date_time"])
test2 = pd.DataFrame(test2)
test1['year'] = test2['date_time'].dt.year
test1['month'] = test2['date_time'].dt.month
test1['day_of_week'] = test2['date_time'].dt.dayofweek
test1['day'] = test2['date_time'].dt.day
test1['hour'] = test2['date_time'].dt.hour
test1["date_time"] = pd.to_datetime(test1["date_time"], format='%Y-%m-%d', errors="coerce")
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["search_span"] = (test1["srch_ci"] - test1["date_time"]).astype('timedelta64[D]')
test1.ix[(test1['hour'] >= 10) & (test1['hour'] < 18), 'hour'] = 1
test1.ix[(test1['hour'] >= 18) & (test1['hour'] < 22), 'hour'] = 2
test1.ix[(test1['hour'] >= 22) & (test1['hour'] == 24), 'hour'] = 3
test1.ix[(test1['hour'] >= 1) & (test1['hour'] < 10), 'hour'] = 3
test1 = test1.drop('srch_ci', axis=1)
test1 = test1.drop('srch_co', axis=1)
test1 = test1.drop('date_time', axis=1)
test1.fillna(-1, inplace=True)
test1.info()
metrics.mapk([[l] for l in test1["stay_span"]], preds, k=5)