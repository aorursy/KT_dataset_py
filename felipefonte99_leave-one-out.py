# We import the libraries

import numpy as np

import pandas as pd

pd.options.display.precision = 15

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import LeaveOneOut, train_test_split

import warnings

warnings.filterwarnings("ignore")
# We load the train.csv

PATH="../input/"

train_df = pd.read_csv(PATH + 'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

train_df.head()
print("There are {} rows in train_df".format(len(train_df)))
# We define the number of rows in each segment as the same number of rows in the real test segments (150000 rows)

rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
train_X.head(3)
train_y.head(3)
def create_features(seg_id, xc, X):

    

    x_roll_std_50 = xc.rolling(50).std().dropna().values

    x_roll_mean_50 = xc.rolling(50).mean().dropna().values

    x_roll_std_1000 = xc.rolling(1000).std().dropna().values

    x_roll_mean_1000 = xc.rolling(1000).mean().dropna().values

    x_roll_std_5000 = xc.rolling(5000).std().dropna().values

    x_roll_mean_5000 = xc.rolling(5000).mean().dropna().values

    

    X.loc[seg_id, 'q05_roll_std_50'] = np.quantile(x_roll_std_50, 0.05)

    X.loc[seg_id, 'q90_roll_mean_50'] = np.quantile(x_roll_mean_50, 0.90)

    X.loc[seg_id, 'q05_roll_std_1000'] = np.quantile(x_roll_std_1000, 0.05)

    X.loc[seg_id, 'q90_roll_mean_1000'] = np.quantile(x_roll_mean_1000, 0.90)

    X.loc[seg_id, 'q05_roll_std_5000'] = np.quantile(x_roll_std_5000, 0.05)

    X.loc[seg_id, 'q90_roll_mean_5000'] = np.quantile(x_roll_mean_5000, 0.90)

    

    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()

    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)

    X.loc[seg_id, 'abs_q75'] = np.quantile(np.abs(xc), 0.75)

    X.loc[seg_id, 'abs_q25'] = np.quantile(np.abs(xc), 0.25)

    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
# iterate over all segments

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows + rows]

    create_features(seg_id, seg['acoustic_data'], train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
train_X.shape, train_y.shape
train_X.head(3)
train_y.head(3)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.25, random_state=42)
seed = 42

    

params_lgb = {'bagging_fraction': 0.9,

            'bagging_freq': 5,

            'early_stopping_rounds': 10,

            'feature_fraction': 0.7,

            'learning_rate': 0.02,

            'num_boost_round': 100,

            'seed': seed,

            'feature_fraction_seed': seed,

            'bagging_seed': seed,

            'drop_seed': seed,

            'data_random_seed': seed,

            'objective': 'regression_l1',

            'boosting': 'gbdt',

            'verbosity': -1,

            'metric': 'mae',

            'num_threads': 8}
# Leave One Out cross validation

loo = LeaveOneOut()

n_splits = loo.get_n_splits(X_train)

t = tqdm_notebook(total=n_splits)

oof = np.zeros(len(X_train))

predictions = np.zeros(len(X_test))



for trn_idx, val_idx in loo.split(X_train):

    t.update(1)

    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

    

    clf = lgb.train(params_lgb, trn_data, valid_sets = [trn_data, val_data], verbose_eval=False)

    oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / n_splits

    

print('CV MAE: {}'.format(mean_absolute_error(y_train.values, oof)))

print('Test MAE: {}'.format(mean_absolute_error(y_test.values, predictions)))
# Scatter plot the real time to failure vs predicted (Leave One Out Cross Validation)

plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), oof)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.title('Leave One Out Cross Validation')

plt.xlabel('time to failure', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
# Scatter plot the real time to failure vs predicted (Testing Set)

plt.figure(figsize=(6, 6))

plt.scatter(y_test.values.flatten(), predictions)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.title('Testing Set')

plt.xlabel('time to failure', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])