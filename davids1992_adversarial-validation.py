from tqdm import tqdm

import time



import numpy as np

import pandas as pd



import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score

from sklearn.model_selection import StratifiedKFold

import tsfresh.feature_extraction.feature_calculators as tff



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
def extract_features(X_all, x, seg_id):

    

    X_all.loc[seg_id, 'ave'] = x.mean()

    X_all.loc[seg_id, 'std'] = x.std()

    X_all.loc[seg_id, 'max'] = x.max()

    X_all.loc[seg_id, 'min'] = x.min()

    X_all.loc[seg_id, 'abs_energy'] = np.dot(x, x)

    X_all.loc[seg_id, 'sum_of_reoccurring_data_points'] = tff.sum_of_reoccurring_data_points(x)

    X_all.loc[seg_id, 'sum_of_reoccurring_values'] = tff.sum_of_reoccurring_values(x)

    X_all.loc[seg_id, 'count_above_mean'] = tff.count_above_mean(x)



    return X_all
rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train_data = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min', 'abs_energy'])

ttf_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



for seg_id in tqdm(range(segments)):

    seg = train.iloc[seg_id*rows:seg_id*rows+rows]

    x = seg['acoustic_data'].values.astype(float)

    ttf_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]



    X_train_data = extract_features(X_train_data, x, seg_id)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test_data = pd.DataFrame(columns=X_train_data.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm(X_test_data.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv', dtype={'acoustic_data': np.int16})

    

    x = seg['acoustic_data'].values.astype(float)

    X_test_data = extract_features(X_test_data, x, seg_id)
X = pd.concat([X_train_data, X_test_data])

y = np.append(np.zeros((X_train_data.shape[0], )), np.ones(X_test_data.shape[0], ))
n_fold = 5

kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
lgb_classifier_params = {'num_leaves': 100,

                         'min_data_in_leaf': 120,

                         'objective': 'binary',

                         'max_depth': -1,

                         'learning_rate': 0.1,

                         "boosting": "gbdt",

                         "metric": 'auc',

                         "verbosity": -1,

                         }
f1s_valid = []

f1s_train = []

all_correctly_recognized = []

for fold_n, (train_index, valid_index) in enumerate(kf.split(X, y)):

    print('\nFold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    model = lgb.LGBMClassifier(**lgb_classifier_params, n_estimators=10000, n_jobs=-1)

    model.fit(X_train, y_train, 

              eval_set=[(X_train, y_train), (X_valid, y_valid)],

              verbose=100, early_stopping_rounds=200)



    y_pred_valid = np.where(model.predict(X_valid, num_iteration=model.best_iteration_) > 0.5, 1, 0)

    y_pred_train = np.where(model.predict(X_train, num_iteration=model.best_iteration_) > 0.5, 1, 0)



    f1s_valid.append(f1_score(y_valid, y_pred_valid))

    f1s_train.append(f1_score(y_train, y_pred_train))
print('CV mean train score: {0:.4f}, std: {1:.4f}.'.format(np.mean(f1s_train), np.std(f1s_train)))
print('CV mean valid score: {0:.4f}, std: {1:.4f}.'.format(np.mean(f1s_valid), np.std(f1s_valid)))
print(classification_report(y_valid, y_pred_valid))
print(classification_report(y_valid, y_pred_valid))
features_importance = pd.DataFrame(sorted(zip(model.feature_importances_, X_train.columns.values)),  columns=['Value', 'Feature'])



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=features_importance.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()
def compare_splits_feature(column, quantile=1.0):

    

    q = X[column].quantile(quantile)

    plt.figure(figsize=(13, 7))

    plt.title(column)

    plt.ylabel('Value')

    plt.xlabel('Split')

    sns.violinplot(x='split', y=column, data=X[X[column] < q])
X['split'] = y

compare_splits_feature('sum_of_reoccurring_data_points', 0.9)

compare_splits_feature('ave')

compare_splits_feature('count_above_mean')
pointer = np.where(valid_index < len(X_train_data))[0]

valid_ids_from_train_data = valid_index[pointer]

X_valid_from_train_data = X_train_data.iloc[valid_ids_from_train_data, :]

correct = y_valid[pointer] == y_pred_valid[pointer]

correctly_recognized = np.where(correct == True)[0]
recognized = np.zeros((len(X_valid_from_train_data), ))

recognized[correctly_recognized] = 1

X_valid_from_train_data['correctly_recognized'] = recognized

X_valid_from_train_data['time_to_failure'] = ttf_train.iloc[valid_ids_from_train_data, :].values

plt.figure(figsize=(13, 7))

plt.title('Correctly recognized training examples vs time_to_failure')

plt.ylabel('Value')

plt.xlabel('Split')

sns.violinplot(x='correctly_recognized', y='time_to_failure',

               data=X_valid_from_train_data, ) #scale='count'

plt.xticks([0, 1], ['False', 'True'])