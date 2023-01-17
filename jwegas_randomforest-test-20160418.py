import numpy as np # linear algebra
import pandas as pd # data processing, CSV f

import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score

import xgboost as xgb
train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
train = pd.DataFrame(columns=train_cols)
train_chunk = pd.read_csv('../input/train.csv', chunksize=100000)
for chunk in train_chunk:
    train = pd.concat( [ train, chunk[chunk['is_booking']==1][train_cols] ] )
train.head()
train_X = train[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
train_y = train['hotel_cluster'].values
#rf = RandomForestClassifier()
#clf = BaggingClassifier(rf)
clf = xgb.XGBClassifier(max_depth=5, n_estimators=10, learning_rate=0.05)
clf.fit(train_X, train_y)
test_y = np.array([])
test_chunk = pd.read_csv('../input/test.csv', chunksize=50000)

for i, chunk in enumerate(test_chunk):
    test_X = chunk[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
    if i > 0:
        test_y = np.concatenate( [test_y, clf.predict_proba(test_X)])
    else:
        test_y = clf.predict_proba(test_X)
    print(i)
def get5Best(x):    
    return " ".join([str(int(z)) for z in x.argsort()[::-1][:5]])
submit = pd.read_csv('../input/sample_submission.csv')
submit['hotel_cluster'] = np.apply_along_axis(get5Best, 1, test_y)
submit.head()
submit.to_csv('submission_20160418_ent_1.csv', index=False)