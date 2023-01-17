import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

import seaborn as sns
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()
# train_df['target'] = train_df['dep_delayed_15min'].map({'Y':1, 'N':0})
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
test_df.head()
train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']

test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']
train_df.head()
train_df.shape, test_df.shape
# from public kernel



# Extract the labels

train_y = train_df.pop('dep_delayed_15min')

train_y = train_y.map({'N': 0, 'Y': 1})



# Concatenate for preprocessing

train_split = train_df.shape[0]

full_df = pd.concat((train_df, test_df))



# Hour and minute

full_df['hour'] = full_df['DepTime'] // 100

full_df.loc[full_df['hour'] == 24, 'hour'] = 0

full_df.loc[full_df['hour'] == 25, 'hour'] = 1

full_df['minute'] = full_df['DepTime'] % 100



# Season

full_df['summer'] = (full_df['Month'].isin(['c-6', 'c-7', 'c-8'])).astype(np.int32)

full_df['autumn'] = (full_df['Month'].isin(['c-9', 'c-10', 'c-11'])).astype(np.int32)

full_df['winter'] = (full_df['Month'].isin(['c-12', 'c-1', 'c-2'])).astype(np.int32)

full_df['spring'] = (full_df['Month'].isin(['c-3', 'c-4', 'c-5'])).astype(np.int32)



# Daytime

full_df['daytime'] = pd.cut(full_df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)

full_df.head()
full_df.drop('DepTime', axis=1, inplace=True)
full_df.head()
# train_df['is_weekend'] = (train_df['DayOfWeek'] == "c-6") | (train_df['DayOfWeek'] == "c-7")

# pd.crosstab(train_df['is_weekend'], train_df['target'])
sns.countplot(data=train_df, x="is_weekend", hue="target");
sns.countplot(data=train_df, x="DayOfWeek", hue="target");
sns.countplot(data=train_df, x="Month", hue="target");
(train_df.groupby('flight')['target'].sum() / train_df.groupby('flight')['target'].count()).sort_values(ascending=False)
train_df[train_df['flight']=='DCA-->ROC']
train_df['DepTime'].describe()
import seaborn as sns



train_df.groupby('target')['Distance'].median()
train_df['hour'] = (train_df['DepTime'] // 100).astype(str)

(train_df.groupby('hour')['target'].sum() / train_df.groupby('hour')['target'].count()).sort_values(ascending=False)
(train_df.groupby('Origin')['target'].sum() / train_df.groupby('Origin')['target'].count()).sort_values(ascending=False)
train_df[ train_df['Origin']=='GST' ]
test_df[ test_df['Origin']=='GST' ]
sns.distplot(train_df['Distance'])
sns.distplot(test_df['Distance'])
train_df.head()
train_df_1 = train_df.drop(['DepTime', 'Origin', 'Dest', 'dep_delayed_15min'], axis=1)

train_df_1.head()
train_df_1.info()
X = train_df_1.drop('target', axis=1)

y = train_df_1['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=17)
X.head()
cf = [f for f in X.columns if f != 'Distance']

cf
from sklearn.preprocessing import OneHotEncoder



one_hot_enc = OneHotEncoder(categorical_features=['hour'], sparse=False)



print('Original number of features: \n', X_train.shape[1], "\n")

data_ohe_train = (one_hot_enc.fit_transform(X_train['hour']))

data_ohe_val = (one_hot_enc.transform(X_val['hour']))

print('Features after OHE: \n', data_ohe_train.shape[1])
from category_encoders import HashingEncoder



train_df = full_df[:train_split]

test_df = full_df[train_split:]
cols = np.array(train_df.columns)

print(cols)

categ_feat_idx = np.where(cols != 'Distance')[0]

print(categ_feat_idx)
train_df['daytime'].dtype
daytime_dict = {'(-0.001, 6.0]':'night', '(6.0, 12.0]':'morning', '(12.0, 18.0]':'afternoon', '(18.0, 23.0]':'evening'}

train_df['daytime_f'] = train_df['daytime'].astype(str).map(daytime_dict)

test_df['daytime_f'] = test_df['daytime'].astype(str).map(daytime_dict)
train_df.head()
train_df.info()
train_df['is_weekend'] = (train_df['DayOfWeek'] == 6) | (train_df['DayOfWeek'] == 7)

test_df['is_weekend'] = (test_df['DayOfWeek'] == 6) | (test_df['DayOfWeek'] == 7)
train_df.head()
train_df['Dep_hour_flag'] = ((train_df['hour'] >= 6) & (train_df['hour'] < 23)).astype('int')

test_df['Dep_hour_flag'] = ((test_df['hour'] >= 6) & (test_df['hour'] < 23)).astype('int')

train_df.head()
tmp_df = pd.read_csv('../input/flight-delays-fall-2018/flight_delays_train.csv')

tmp_df['target'] = tmp_df['dep_delayed_15min'].map({'N':0, 'Y':1})

tmp = tmp_df.groupby('Dest')['target'].sum() / tmp_df.groupby('Dest')['target'].count()

tmp
train_df['busy_dest'] = train_df['Dest'].map(tmp.to_dict())

test_df['busy_dest'] = test_df['Dest'].map(tmp.to_dict())
train_df['busy_day'] = ((train_df['DayOfWeek'] == 1) | (train_df['DayOfWeek'] == 4) | (train_df['DayOfWeek'] == 5)).astype('int')

test_df['busy_day'] = ((test_df['DayOfWeek'] == 1) | (test_df['DayOfWeek'] == 4) | (test_df['DayOfWeek'] == 5)).astype('int')
z = test_df['busy_dest']

del test_df['busy_dest']

test_df['busy_dest'] = z

test_df.head()
X_test = test_df.drop('daytime', axis=1).values
X_train = train_df.drop('daytime', axis=1).values

y_train = train_y.values

X_test = test_df.drop('daytime', axis=1).values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
cols = np.array(train_df.drop('daytime', axis=1).columns)

print(cols)

categ_feat_idx = np.where((cols != 'Distance') & (cols != 'busy_dest'))[0]

print(categ_feat_idx)
params = {'iterations':2000,

          'loss_function':'Logloss',

          'eval_metric':'AUC',

          'cat_features': categ_feat_idx,

          #'ignored_features':[7, 17],

          'task_type': 'GPU',

          'border_count': 32,

          'verbose': 200,

          'random_seed': 17

         }

ctb = CatBoostClassifier(**params)
%%time

ctb.fit(X_train_part, y_train_part);
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]

ctb_valid_pred
roc_auc_score(y_valid, ctb_valid_pred)
%%time

ctb.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')
!head ctb_pred.csv