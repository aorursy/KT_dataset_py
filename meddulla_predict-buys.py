# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import lightgbm as lgb

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular.transform import add_cyclic_datepart

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

PATH = '/kaggle/input/recsys-challenge-2015' 
def read_buys(limit=None):

    buys = pd.read_csv(f"{PATH}/yoochoose-buys.dat",

                    names=["session", "timestamp", "item", "price", "qty"],

                   parse_dates=["timestamp"])

    buys = buys.sort_values(by=["timestamp", "session"])

    if limit:

        buys = buys.iloc[:limit]

    return buys





def read_clicks(limit=None):

    print("Loading clicks")

    filename = f"{PATH}/yoochoose-clicks.dat"

    df = pd.read_csv(filename,

                     names=["session", "timestamp", "item", "category"],

                     parse_dates=["timestamp"],

                     converters={"category": lambda c: -1 if c == "S" else c})

    df = df.sort_values(by=["timestamp", "session"])

    if limit:

        df = df.iloc[:limit]

    print("Clicks shape %s %s" % df.shape)

    return df





def process_clicks(clicks, rolling_days=15):

    # Compute dwell time for each click

    print("Processing clicks")

    clicks['prev_ts'] = clicks.groupby('session')['timestamp'].transform(lambda x: x.shift())

    clicks['diff_prev'] = clicks["timestamp"] - clicks["prev_ts"] # in minutes

    clicks["dwell"] = clicks.groupby('session')['diff_prev'].transform(lambda x: x.shift(-1)).dt.seconds/60

    clicks = clicks.sort_values(by=["session", "timestamp"])

    print("Processed clicks shape %s %s" % clicks.shape)

    return clicks





def process_buys(limit=None):

    # Group into sessions, compute nr of items bought and set label column

    buys = read_buys(limit=limit)

    print("Processing buys")

    print("Buys from %s to %s" % (buys.timestamp.min(), buys.timestamp.max()))

    grouped = buys.groupby("session")

    buys_g = pd.DataFrame(index=grouped.groups.keys())

    buys_g["items_bought"] = grouped.item.count() # quantity may be zero which is weird so dont use it

    buys_g["is_buy"] = 1 # for easier merge later on

    buys_g.index.name = "session"

    print("Buys grouped by session %s %s" % buys_g.shape)

    return buys_g





def get_items_cats_percent(clicks, limit=None):

    buys = read_buys(limit=limit)

    # percent bought

    item_id_bought_pct = buys.item.value_counts(normalize=True)

    cat_id_viewed_pct = clicks.category.value_counts(normalize=True)

    item_id_viewed_pct = clicks.item.value_counts(normalize=True)



    return dict(views=dict(item=item_id_viewed_pct, cat=cat_id_viewed_pct), buys=item_id_bought_pct)





def process_sessions(processed_clicks, limit=None):

    print("Preprocessing - Grouping clicks into sessions")

    clicks = processed_clicks

    

    # Group clicks by session

    grouped = clicks.groupby("session")

    sessions = pd.DataFrame(index=grouped.groups.keys())

    

    # Session counters

    sessions["total_clicks"] = grouped.item.count()

    sessions["total_items"] = grouped.item.unique().apply(lambda x: len(x))

    sessions["total_cats"] = grouped.category.unique().apply(lambda x: len(x))

    print("Computed counters")

    

    # Session duration

    sessions["max_dwell"] = grouped.dwell.max()

    sessions["mean_dwell"] = grouped.dwell.mean()

    sessions["start_ts"] = grouped.timestamp.min()

    sessions["end_ts"] = grouped.timestamp.max()

    sessions["total_duration"] = (sessions["end_ts"] - sessions["start_ts"]).dt.seconds / 60

    print("Computed dwell and duration")

    

    # Click rate

    sessions["total_duration_secs"] = (sessions["end_ts"] - sessions["start_ts"]).dt.seconds

    sessions["click_rate"] = sessions["total_clicks"] / sessions["total_duration_secs"]

    sessions.click_rate = sessions.click_rate.replace(np.inf, np.nan)

    sessions.click_rate = sessions.click_rate.fillna(0)

    del sessions["total_duration_secs"]

    print("Computed click rate")

    

    # Dates

    #sessions = add_datepart(sessions, "start_ts", drop=False)

    #sessions = add_datepart(sessions, "end_ts", drop=False)

    sessions = add_cyclic_datepart(sessions, "start_ts", drop=False)

    sessions = add_cyclic_datepart(sessions, "end_ts", drop=False)

    print("Computed cyclic date parts")

    

    # What is the item and cat most viewed in this session?

    # How many times were they viewed?

    sessions["cat_most_viewed_n_times"] = grouped.category.value_counts().unstack().max(axis=1)

    sessions["cat_most_viewed"] = grouped.category.value_counts().unstack().idxmax(axis=1)

    sessions["item_most_viewed_n_times"] = grouped.item.value_counts().unstack().max(axis=1)

    sessions["item_most_viewed"] = grouped.item.value_counts().unstack().idxmax(axis=1)

    print("Computed most viewed item/cat per session")



    # For the item most viewed in each session, what is its global buy/view frequency?

    freqs = get_items_cats_percent(clicks, limit=limit)

    cat_views = pd.DataFrame(freqs["views"]["cat"])

    cat_views.columns = ["cat_views_freqs"]

    sessions = sessions.merge(cat_views, how="left", left_on="cat_most_viewed", right_index=True)

    sessions.cat_views_freqs = sessions.cat_views_freqs.fillna(0)

    item_views = pd.DataFrame(freqs["views"]["item"])

    item_views.columns = ["item_views_freqs"]

    sessions = sessions.merge(item_views, how="left", left_on="item_most_viewed", right_index=True)

    sessions.item_views_freqs = sessions.item_views_freqs.fillna(0)

    item_buys = pd.DataFrame(freqs["buys"])

    item_buys.columns = ["item_buys_freqs"]

    sessions = sessions.merge(item_buys, how="left", left_on="item_most_viewed", right_index=True)

    sessions.item_buys_freqs = sessions.item_buys_freqs.fillna(0)

    print("Computed most viewed/bought freqs")

    

    # Sorting sessions

    sessions = sessions.sort_values(by=["start_ts"])

    sessions.index.name = "session"

    

    print("Sessions shape %s %s" % sessions.shape)

    print("Sessions columns %s " % sessions.columns)

    print("Sessions from %s to %s" % (sessions.start_ts.min(), sessions.start_ts.max()))

    return sessions





def prep(limit=None):

    print("Prepping data for classification")

    buys = process_buys(limit=limit)

    clicks = read_clicks(limit=limit)

    processed_clicks = process_clicks(clicks)

    sessions = process_sessions(clicks, limit=limit)

    

    print("Merging clicks and buys")

    X = pd.merge(sessions, buys, how="left", left_index=True, right_index=True)

    

    X = X.sort_values(by=["start_ts"])

    y = X["is_buy"]

    y = y.fillna(0)

    

    X["cat_most_viewed"] = X["cat_most_viewed"].astype("float64")

    

    # Delete label

    del X["is_buy"]

    

    # Delete time columns (cant be used as is and we already have the cyclic date parts)

    del X["start_ts"]

    del X["end_ts"]

    

    return X, y



def classify(X, y):

    

    print("Splitting into train and test")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25, shuffle=False)



    params = {

        'boosting_type': 'gbdt', 

        'objective': 'binary', 'metric': 'auc',

        'learning_rate': 0.03, 'num_leaves': 7, 'max_depth': 3,

        'min_child_samples':100, # min_data_in_leaf

        'max_bin': 100, #number of bucked bin for feature values

        'subsample': 0.9, # subsample ratio of the training instance

        'subsample_freq':1, # frequence of subsample

        'colsample_bytree': 0.7, # subsample ratio of columns when constructing each tree.

        'min_child_weight':0,

        'min_split_gain':0, # lambda_l1, lambda_l2 and min_gain_to_split to regularization.

        'nthread':8, 'verbose': 0, 

        'scale_pos_weight': 150 # because training data is extremely unbalanced

    }

    

    print("Building datasets for lightgbm")

    # prepare model

    dtrain = lgb.Dataset(X_train,label=y_train,feature_name=X.columns.tolist())

    dvalid = lgb.Dataset(X_test,label=y_test,feature_name=X.columns.tolist())

    

    cats = ["cat_most_viewed", "item_most_viewed"]

    evals_results = {}

    

    print("Starting classification")

    model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid],

                      #categorical_feature=cats,

                      valid_names=['train', 'valid'],

                      evals_result = evals_results, num_boost_round=1000,

                      #early_stopping_rounds= 100, 

                      verbose_eval=50, feval = None)

    

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_pred = np.round_(y_pred, 0)

    print('The accuracy of prediction is:', metrics.accuracy_score(y_test, y_pred))

    print('The roc_auc_score of prediction is:', metrics.roc_auc_score(y_test, y_pred))

    print('The null acccuracy is:', max(y_test.mean(), 1 - y_test.mean()))

    

    return model
X, y = prep(limit=100000)
model = classify(X, y)

vs = model.feature_importance()

ks = X.columns

d = dict(zip(ks, vs))

sorted(d.items(), key=lambda x: x[1], reverse=True)[:15]