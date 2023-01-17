import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import pandas as pd

click_data = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train_sample.csv',parse_dates=['click_time'])
click_data.head()
print("Shape of click_data is : {}".format(click_data.shape))
# Add new columns for timestamp features day, hour, minute, and second

clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')
clicks.head()
from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']
lable_encoder = preprocessing.LabelEncoder()

for feature in cat_features:
    encoded = lable_encoder.fit_transform(clicks[feature])
    clicks[feature +'_labels'] = encoded
clicks.head()
feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[:-valid_rows * 2]
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]
train.head()
valid.head()
test.head()
# msno.bar(train)

plt.style.use('seaborn-colorblind')
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

msno.bar(train, ax = ax1, color=(10/255, 3/255, 250/255), fontsize=10)
msno.bar(test, ax = ax2, color=(251/255, 0/255, 0/255), fontsize=10)

ax1.set_title('Train Missing Values Map', fontsize = 16)
ax2.set_title('Test Missing Values Map', fontsize = 16);
plt.style.use('dark_background')
plt.figure(figsize=(10,8))  
# sns.set(style="darkgrid")
ax = sns.countplot(x = train['is_attributed'])
plt.style.use('dark_background')
plt.figure(figsize=(10,8))  
# sns.set(style="darkgrid")
ax = sns.countplot(x = valid['is_attributed'])
plt.style.use('dark_background')
plt.figure(figsize=(10,8))  
# sns.set(style="darkgrid")
ax = sns.countplot(x = test['is_attributed'])
import lightgbm as lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=3)
from sklearn import metrics

ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')
def get_data_splits(dataframe, valid_fraction=0.1):
    """Splits a dataframe into train, validation, and test sets.

    First, orders by the column 'click_time'. Set the size of the 
    validation and test sets with the valid_fraction keyword argument.
    """

    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]

    valid = dataframe[-valid_rows * 2:-valid_rows]
    test = dataframe[-valid_rows:]
    
    return train, valid, test

def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                           'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    
    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 
                    early_stopping_rounds=20, verbose_eval=False)
    
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    
    if test is not None: 
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score
print("Baseline model")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)
import category_encoders as ce

cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_data_splits(clicks)
import category_encoders as ce

count_enc = ce.CountEncoder(cols=cat_features)

count_enc.fit(train[cat_features])

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix("_count"))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix("_count"))
_ = train_model(train_encoded, valid_encoded)
import category_encoders as ce


target_enc = ce.TargetEncoder(cols = cat_features)

target_enc.fit(train[cat_features], train['is_attributed'])

train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix("_target"))

valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix("_target"))
# train_encoded = train_encoded.drop(['ip_target'],axis = 1)
# valid_encoded = valid_encoded.drop(['ip_target'],axis = 1)
_ = train_model(train_encoded, valid_encoded)

cat_features = ['app', 'device', 'os', 'channel']

cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

cb_enc.fit(train[cat_features], train['is_attributed'])

train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))
_ = train_model(train_encoded, valid_encoded)
import itertools

cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features

for c1, c2 in itertools.combinations(cat_features,2):
    new_col = '_'.join([c1,c2])
    values = clicks[c1].map(str) + "_" + clicks[c2].map(str)
    encoder = preprocessing.LabelEncoder()
    interactions[new_col] = encoder.fit_transform(values)
clicks = clicks.join(interactions)
print("Score with interactions")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)
interaction_features = ['ip_app','ip_device', 'ip_os', 'ip_channel', 'app_device', 'app_os','app_channel', 'device_os', 
                        'device_channel', 'os_channel']
clicks[interaction_features].head()
def count_past_events(series):
    activity = pd.Series(series.index, index = series, name="past_6_hours").sort_index()
    past_6_hours = activity.rolling('6H').count() - 1
    return past_6_hours
past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')
clicks['ip_past_6hr_counts'] = past_events

train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)
import matplotlib.pyplot as plt

plt.plot(clicks.ip_past_6hr_counts[6:]);
plt.title("Activity of ip within 6 hrs");
def time_diff(series):
    """Returns a series with the time since the last timestamp in seconds."""
    return series.diff().dt.total_seconds()
past_events = pd.read_parquet('../input/feature-engineering-data/time_deltas.pqt')
clicks['past_events_6hr'] = past_events

train, valid, test = get_data_splits(clicks.join(past_events))
_ = train_model(train, valid)
def previous_attributions(series):
    """Returns a series with the number of times an app has been downloaded."""
    sums = series.expanding(min_periods=2).sum() - series
    return sums
past_events = pd.read_parquet('../input/feature-engineering-data/downloads.pqt')
clicks['ip_past_6hr_counts'] = past_events

train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)
test = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv',
                         parse_dates=['click_time'])
test.head()
# feature_cols = ['day', 'hour', 'minute', 'second', 
#                 'ip_labels', 'app_labels', 'device_labels',
#                 'os_labels', 'channel_labels']

test_x = test.copy()
test_x['day'] = test_x['click_time'].dt.day.astype('uint8')
test_x['hour'] = test_x['click_time'].dt.hour.astype('uint8')
test_x['minute'] = test_x['click_time'].dt.minute.astype('uint8')
test_x['second'] = test_x['click_time'].dt.second.astype('uint8')
from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']
lable_encoder = preprocessing.LabelEncoder()

for feature in cat_features:
    encoded = lable_encoder.fit_transform(test_x[feature])
    test_x[feature +'_labels'] = encoded
test_x.head()
test_x = test_x.drop(["click_time","ip","channel","click_id","app","device","os"], axis = 1)
test_x.head()
predictions = bst.predict(test_x)
print(predictions.shape)

predict = np.array(predictions)
predict = np.around(predict,decimals = 0)
data = {
    "click_id": test.click_id,
    "is_attributed": predict
}
sub = pd.DataFrame(data = data)
sub['is_attributed'] = sub['is_attributed'].astype(int)
# sub.to_csv("submission.csv",index = False)
sub.head()