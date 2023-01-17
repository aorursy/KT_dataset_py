

%matplotlib inline



import itertools

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics



ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched'])



# Drop live projects

ks = ks.query('state != "live"')



# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))



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



def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test



def train_model(train, valid):

    feature_cols = train.columns.drop('outcome')



    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])



    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    print("Training model!")

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)

    print(f"Validation AUC score: {valid_score:.4f}")

    return bst
from sklearn.feature_selection import SelectKBest, f_classif



feature_cols = baseline_data.columns.drop('outcome')



# Keep 5 features

selector = SelectKBest(f_classif, k=5)



X_new = selector.fit_transform(baseline_data[feature_cols], baseline_data['outcome'])

X_new
feature_cols = baseline_data.columns.drop('outcome')

train, valid, _ = get_data_splits(baseline_data)



# Keep 5 features

selector = SelectKBest(f_classif, k=5)



X_new = selector.fit_transform(train[feature_cols], train['outcome'])

X_new
# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train.index, 

                                 columns=feature_cols)

selected_features.head()
# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

valid[selected_columns].head()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



train, valid, _ = get_data_splits(baseline_data)



X, y = train[train.columns.drop("outcome")], train['outcome']



# Set the regularization parameter C=1

logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=7).fit(X, y)

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(X)

X_new
# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=X.index,

                                 columns=X.columns)



# Dropped columns have values of all 0s, keep other columns 

selected_columns = selected_features.columns[selected_features.var() != 0]