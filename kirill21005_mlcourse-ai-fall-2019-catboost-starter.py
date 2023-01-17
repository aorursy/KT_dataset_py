import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()
train_df['target'] = train_df['dep_delayed_15min'].map({'Y':1, 'N':0})

train_df.head()
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
test_df.head()
train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']

test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']
(train_df.groupby('flight')['target'].sum()/train_df.groupby('flight')['target'].count()).sort_values(ascending =False)
import seaborn as sns
train_df.groupby('target')['Distance'].median()
train_df['hour'] = train_df['DepTime']//100

test_df['hour'] = test_df['DepTime']//100



#(train_df.groupby('hour')['target'].sum()/train_df.groupby('hour')['target'].count()).sort_values(ascending =False)
train_df['Month'] = train_df['Month'].str.replace(r'\D', '')

train_df['DayofMonth'] = train_df['DayofMonth'].str.replace(r'\D', '')

train_df['DayOfWeek'] = train_df['DayOfWeek'].str.replace(r'\D', '')

test_df['Month'] = train_df['Month'].str.replace(r'\D', '')

test_df['DayofMonth'] = train_df['DayofMonth'].str.replace(r'\D', '')

test_df['DayOfWeek'] = train_df['DayOfWeek'].str.replace(r'\D', '')



train_df.head()


train_df.loc[train_df['hour'] == 24, 'hour'] = 0

train_df.loc[train_df['hour'] == 25, 'hour'] = 1

train_df['minute'] = train_df['DepTime'] % 100

test_df.loc[test_df['hour'] == 24, 'hour'] = 0

test_df.loc[test_df['hour'] == 25, 'hour'] = 1

test_df['minute'] = test_df['DepTime'] % 100



# Season

train_df['summer'] = (train_df['Month'].isin(['6', '7', '8'])).astype(np.int32)

train_df['autumn'] = (train_df['Month'].isin(['9', '10', '11'])).astype(np.int32)

train_df['winter'] = (train_df['Month'].isin(['12', '1', '2'])).astype(np.int32)

train_df['spring'] = (train_df['Month'].isin(['3', '4', '5'])).astype(np.int32)

test_df['summer'] = (test_df['Month'].isin(['6', '7', '8'])).astype(np.int32)

test_df['autumn'] = (test_df['Month'].isin(['9', '10', '11'])).astype(np.int32)

test_df['winter'] = (test_df['Month'].isin(['12', '1', '2'])).astype(np.int32)

test_df['spring'] = (test_df['Month'].isin(['3', '4', '5'])).astype(np.int32)

train_df.head()
test_df.head()
temp = train_df.groupby('UniqueCarrier')['Distance'].sum()



train_df['Summa'] = train_df['UniqueCarrier'].map(temp.to_dict())



temptest = test_df.groupby('UniqueCarrier')['Distance'].sum()

test_df['Summa'] = test_df['UniqueCarrier'].map(temptest.to_dict())

test_df.head()
test_df.drop('DepTime',axis =1)

train_df.drop('DepTime',axis =1)
train_df.drop('DepTime', axis =1)
sns.distplot(train_df['Distance'])
train_df = train_df.drop('target', axis=1)
categ_feat_idx = np.array([0,1,2,3,4,5,6,8,9,10,11,12,13,14,15])

categ_feat_idx
X_train = train_df.drop('dep_delayed_15min', axis=1).values

y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

X_test = test_df.values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
ctb1 = CatBoostClassifier(random_seed=17, silent=True)
%%time

ctb1.fit(X_train_part, y_train_part,

        cat_features=categ_feat_idx);
ctb1_valid_pred = ctb1.predict_proba(X_valid)[:, 1]

ctb1_valid_pred.sum()
roc_auc_score(y_valid, ctb1_valid_pred)
%%time

ctb1.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_pred = ctb1.predict_proba(X_train)[:, 1]

roc_auc_score(y_train, ctb_pred)
ctb_test_pred = ctb1.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_test_pred.csv')
!head ctb_test_pred.csv