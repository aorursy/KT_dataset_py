# Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.random_projection import *

np.random.seed(0)

# Utils functions
def convert_totime(seconds):
    return datetime.datetime.fromtimestamp(seconds)

def add_srp_features(df, n_comp, features):
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=17)
    srp_results = srp.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_srp_' + str(i)] = srp_results[:, i - 1]
    return df
# Load data
df = pd.read_csv('../input/creditcard.csv', header=0)
features = [f for f in list(df) if "V" in f]

# Feature engineering: Time
df['datetime'] = df.Time.apply(convert_totime)
df['hour'] = df.datetime.dt.hour
df['minute'] = df.datetime.dt.minute

# Feature engineering: Projections
df = add_srp_features(df, 3, features)
CV_NUMBER = 10

excl = ["Class", "datetime"]
features = [f for f in df.columns if f not in excl]

auc_list = []
recall_list = []

# Cross Validation
for idx in range(CV_NUMBER):

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['Class'], test_size=0.1, random_state=7*idx)

    # Create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # Params
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.04,
        'bagging_fraction': 0.95,
        'feature_fraction': 0.98,
        'bagging_freq': 6,
        'max_depth': -1,
        'max_bin': 511,
        'min_data_in_leaf': 20,
        'verbose': 0,
        'seed': 23747 + 17 * idx
    }
    
    bst = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=lgb_eval, early_stopping_rounds=20)    
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred = np.round_(y_pred, 0)

    auc_list.append(roc_auc_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
# Mean Metric
print(('Area Under the Curve mean: {}'.format(np.mean(auc_list))))
print(('Recall mean: {}'.format(np.mean(recall_list))))