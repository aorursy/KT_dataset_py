import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import json



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor



sns.set_style('darkgrid')
# train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv',nrows=22345) # read in sample of data for fast experimentation



train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')

sample = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')

with open('../input/bigquery-geotab-intersection-congestion/submission_metric_map.json') as f:

    submission_metric_map = json.load(f)
train.head()
test.head()
train.info()
test.info()
street_features = ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path']

train[street_features].head()
train.drop('Path', axis=1, inplace=True)

test.drop('Path', axis=1, inplace=True)
directions = {

    'N': 0,

    'NE': 1/4,

    'E': 1/2,

    'SE': 3/4,

    'S': 1,

    'SW': 5/4,

    'W': 3/2,

    'NW': 7/4

}
train['EntryHeading'] = train['EntryHeading'].map(directions)

train['ExitHeading'] = train['ExitHeading'].map(directions)



test['EntryHeading'] = test['EntryHeading'].map(directions)

test['ExitHeading'] = test['ExitHeading'].map(directions)
train['diffHeading'] = train['EntryHeading']-train['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration



test['diffHeading'] = test['EntryHeading']-test['ExitHeading']  # TODO - check if this is right. For now, it's a silly approximation without the angles being taken into consideration



train[['ExitHeading','EntryHeading','diffHeading']].drop_duplicates().head(10)
new_train_columns = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',

       'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'DistanceToFirstStop',

       'Month', 'TotalTimeStopped', 'Percentile', 'City'

                    ,'diffHeading'

                    ]
new_test_columns = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',

       'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',

       'Month', 'Percentile', 'City'

                   ,'diffHeading'

                   ]
new_train = pd.DataFrame(columns=new_train_columns)
new_test = pd.DataFrame(columns=new_test_columns)
for per in [20, 40, 50, 60, 80]:

    new_df = train.copy()

    new_df['TotalTimeStopped'] = new_df['TotalTimeStopped_p'+str(per)]

    new_df['DistanceToFirstStop'] = new_df['DistanceToFirstStop_p'+str(per)]

    new_df['Percentile'] = pd.Series([per for _ in range(len(new_df))])

    new_df.drop(['TotalTimeStopped_p20', 'TotalTimeStopped_p40',

       'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',

       'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',

       'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',

       'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',

       'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',

       'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80', 'RowId'], axis=1,inplace=True)

    new_train = pd.concat([new_train, new_df], sort=True)
for per in [20, 50, 80]:

    new_df = test.copy()

    new_df['Percentile'] = pd.Series([per for _ in range(len(new_df))])

    new_test = pd.concat([new_test, new_df], sort=True)
new_train = pd.concat([new_train.drop('City', axis=1), pd.get_dummies(new_train['City'])], axis=1)
new_test = pd.concat([new_test.drop('City', axis=1), pd.get_dummies(new_test['City'])], axis=1)
new_train = new_train.reindex(sorted(new_train.columns), axis=1)

new_test = new_test.reindex(sorted(new_test.columns), axis=1)
new_test = new_test.sort_values(by=['RowId', 'Percentile'])
X_train = np.array(new_train.drop(['EntryStreetName', 'ExitStreetName', 'IntersectionId', 'TotalTimeStopped', 'DistanceToFirstStop'], axis=1))

X_test = np.array(new_test.drop(['EntryStreetName', 'ExitStreetName', 'IntersectionId', 'RowId'], axis=1))
y_train = np.array(new_train[['TotalTimeStopped', 'DistanceToFirstStop']])
X_train.shape
X_test.shape
clf = RandomForestRegressor(n_jobs=4, n_estimators=12)



clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
sample['Target'] = y_pred.reshape(-1)
l = []

for i in range(1920335):

    for j in [0,3,1,4,2,5]:

        l.append(str(i)+'_'+str(j))
sample['TargetId'] = l
sample = sample.sort_values(by='TargetId')
sample.to_csv('sample_submission.csv', index=False)