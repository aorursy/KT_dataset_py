import numpy as np

import pandas as pd

import seaborn as sns



import os

print(os.listdir("../input"))



from sklearn.model_selection import ShuffleSplit



import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("train.shape", train.shape, "test shape", test.shape)
train.dtypes
train.head()
train.info()
train.isna().sum()
train.trip_duration.min()
train.trip_duration.max()
train.hist(bins=50, figsize=(20,15), color="#F3A111")

plt.show()
train.loc[train['trip_duration'] < 5000, 'trip_duration'].hist(color="#F3A111");
plt.subplots(figsize=(18,7))

plt.title("Visualisation des outliers")

train.boxplot();
train.loc[train['trip_duration'] < 3700, 'trip_duration'].hist(color="#F3A111");
train = train[train['trip_duration']<= 3700]
train = train[(train['trip_duration'] > 60) & (train['trip_duration'] < 3600 * 24)]



train['hour'] = train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



train['distance'] = np.sqrt((train['pickup_latitude']-train['dropoff_latitude'])**2

                        + (train['pickup_longitude']-train['dropoff_longitude'])**2)





test['hour'] = test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))



test['distance'] = np.sqrt((test['pickup_latitude']-test['dropoff_latitude'])**2

                        + (test['pickup_longitude']-test['dropoff_longitude'])**2)



train.shape, test.shape
train.isnull().sum()
difference_col = list(set(train.columns).difference(set(test.columns)))
y_train = train["trip_duration"] # My target

X_train = train[["pickup_longitude","passenger_count", "pickup_latitude", "dropoff_longitude","dropoff_latitude","distance","hour"]] # My features



X_datatest = test[["pickup_longitude","passenger_count", "pickup_latitude", "dropoff_longitude","dropoff_latitude","distance","hour"]]
train.drop(['hour','distance']+difference_col, axis=1, inplace=True)
#grd = SGDRegressor()



grd = SGDRegressor()

grd.fit(X_train, y_train)
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
rff = RandomForestRegressor(n_estimators=10, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=80, bootstrap=True)

rff.fit(X_train, y_train)

rff.score(X_valid, y_valid)
# calculate the cross validation scores of the model



cv = ShuffleSplit(n_splits=4, test_size=0.8, random_state=42)

cv_scores = cross_val_score(rff, X_train, y_train, cv=cv, scoring= 'neg_mean_squared_log_error')
cv_scores
for i in range(len(cv_scores)):

    cv_scores[i] = np.sqrt(abs(cv_scores[i]))

print(np.mean(cv_scores))
train_pred = rff.predict(X_datatest)
train_pred
my_submission = pd.DataFrame({'id': test.id, 'trip_duration': train_pred})

my_submission.head()
my_submission.to_csv('submission.csv', index=False)