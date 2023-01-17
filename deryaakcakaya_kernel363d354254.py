# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import itertools

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn.feature_selection import SelectKBest, f_classif



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter







# read csv.

ks = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", parse_dates=['deadline', 'launched'])

# add count.

ks['count'] = 1



# add state_type.

ks.loc[ks['state'] == 'failed', 'state_type'] = 'failed'

ks.loc[ks['state'] == 'successful', 'state_type'] = 'successful'



# to datetime.

ks['launched'] = pd.to_datetime(ks['launched'])

ks['deadline'] = pd.to_datetime(ks['deadline'])



# add days.

ks['days'] = (ks['deadline'] - ks['launched']).dt.days



ks_failed = ks[ks['state'] != 'successful']

ks_successful = ks[ks['state'] == 'successful']





#success

ks2 = ks_successful[['launched', 'count']]

ks2.set_index('launched', inplace=True)

ks2 = ks2.resample('Y').sum() 

ks2



# failed

ks3 = ks_failed[['launched', 'count']]

ks3.set_index('launched', inplace=True)

ks3 = ks3.resample('Y').sum() 

ks3



ks4 = ks[['launched', 'count']]

ks4 = ks4[(ks4['launched'] >= '2009-01-01') & (ks4['launched'] < '2018-01-01')]

ks4.set_index('launched', inplace=True)

ks4 = ks4.resample('Y').sum() 

ks4



ks5 = ks2 / ks4 * 100



ks5['count'].plot(label="Success rate", figsize = (8, 6))

plt.title('Success rate Trend')

plt.legend(ncol=1)

plt.show()



# Drop live projects

ks = ks.query('state != "live"')



# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))



ks.head()
# Timestamp features

ks = ks.assign(hour=ks.launched.dt.hour,

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)



# Label encoding

cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()

encoded = ks[cat_features].apply(encoder.fit_transform)



data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']

baseline_data = ks[data_cols].join(encoded)



cat_features = ['category', 'currency', 'country']

interactions = pd.DataFrame(index=ks.index)

for col1, col2 in itertools.combinations(cat_features, 2):

    new_col_name = '_'.join([col1, col2])

    # Convert to strings and combine

    new_values = ks[col1].map(str) + "_" + ks[col2].map(str)

    label_enc = LabelEncoder()

    interactions[new_col_name] = label_enc.fit_transform(new_values)

baseline_data = baseline_data.join(interactions)



baseline_data.head()
launched = pd.Series(ks.index, index=ks.launched, name="count_7_days").sort_index()

count_7_days = launched.rolling('7d').count() - 1

count_7_days.index = launched.values

count_7_days = count_7_days.reindex(ks.index)



baseline_data = baseline_data.join(count_7_days)



def time_since_last_project(series):

    # Return the time in hours

    return series.diff().dt.total_seconds() / 3600.



df = ks[['category', 'launched']].sort_values('launched')

timedeltas = df.groupby('category').transform(time_since_last_project)

timedeltas = timedeltas.fillna(timedeltas.max())



baseline_data = baseline_data.join(timedeltas.rename({'launched': 'time_since_last_project'}, axis=1))



baseline_data.head()
def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test



train, valid, test = get_data_splits(baseline_data)

train.head()
def train_model(train, valid):

    feature_cols = train.columns.drop('outcome')



    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])



    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    print("Training model!")

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    # Try to predict outcome

    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)

    print(f"Validation AUC score: {valid_score:.8f}")

    return bst



train_model(train, valid)
feature_cols = baseline_data.columns.drop('outcome')

train, valid, _ = get_data_splits(baseline_data)



# Keep 5 features

selector = SelectKBest(f_classif, k=5)



X_new = selector.fit_transform(train[feature_cols], train['outcome'])



# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train.index, 

                                 columns=feature_cols)





# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]



selected_columns
train_model(train, valid)



# Get the valid dataset with the selected features.

valid[selected_columns].head()