import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score

# Taking care of data leak:
# Match test rows from train by user_location_country, 
#                               user_location_region, 
#                               user_location_city, 
#                               hotel_market, 
#                               orig_destination_distance and get hotel_clusters for free!

# 1: Create a dataset that is able to train a classifier correctly.
#    The problem here is that even though we could train the classifier using the clicks and bookings,
#    the test set has no click events so the information learned there could be useless and misleading.
#    We have to take into account that using the clicks could teach us the way to recognize
#    what the user do not want to book, but that information could be marginal when the aim is to 
#    find the hotel they actually booked.
#    Investigate the option of formatting the train set in a way that mirrors the test set by
#    putting all the information in one row.
#
# 2: Train a classifier suitable to the problem.
#    We have to use a classifier that gives class probabilities in the end in order to find
#    the top 5 most probable hotel_clusters.
#
# 3: Data leak.
#    After fitting and evaulating the classifier(s), take care of the data leak oulined above.
#
# 4: DO IT AGAIN.
#    Don't get discouraged as always. 6 weeks are a lot. You usually give up after one week and the 
#    first few unsuccessful attempts to get a better score you lazy scumbag. You want to become a 
#    full-stack data scientist, so behave like one.
#
# 5: Have a beer and congratulate yourself on the hard work you have done during these 46 days.
#    Unless you gave up after 2 weeks you fucker. Then try again next time you bastard.


# 1: Create a dataset that is able to train a classifier correctly.
#    First instinct: Remove the clicks, keep the bookings. 
#    FROM: RandomForest_test_20160418 by Evgenii Zhukov

train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
train = pd.DataFrame(columns=train_cols)
train_chunk = pd.read_csv('../input/train.csv', chunksize=100000)

for chunk in train_chunk:
    train = pd.concat( [ train, chunk[chunk['is_booking']==1][train_cols] ] )
    
train.head()
train_X = train[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
train_y = train['hotel_cluster'].values

# 2: Train a classifier suitable to the problem.
#    FROM: RandomForest_test_20160418 by Evgenii Zhukov

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=2016, n_jobs=4)
clf = BaggingClassifier(rf, n_estimators=2, max_samples=0.1, random_state=2014, n_jobs=4)
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
submit.to_csv('Zhukov_0425.csv', index=False)

# 3: Data leak.


