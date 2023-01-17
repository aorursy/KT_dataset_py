import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



print('Train dataset size:', train_df.shape)

print('Test dataset size:', test_df.shape)
train_df.head(10)
f_cols = [f'var_{i}' for i in range(200)]
f, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

ax = ax.flatten()



for i in range(len(ax)):

    col1 = train_df[f_cols[i]].values

    col2 = test_df[f_cols[i]].values

    

    ax[i].hist(col1, bins=100)

    ax[i].hist(col2, bins=100, alpha=0.5)

    ax[i].set_title(f_cols[i])
feature_means = np.mean(train_df[f_cols].values, axis=0)

plt.hist(feature_means, bins=10)

_ = plt.title('feature mean values')
train_features = train_df[f_cols]

corr_matrix = train_features.corr()

plt.subplots(figsize=(20, 10))

_ = sns.heatmap(corr_matrix, vmax=1, square=True)

_ = plt.title('Correlation matrix')
from sklearn.model_selection import train_test_split



X = train_df[f_cols].values

Y = train_df['target'].values



train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
from catboost import CatBoostClassifier



def getModel():

    model = CatBoostClassifier(

        task_type='GPU',

        iterations=20000,

        learning_rate=0.05,

        l2_leaf_reg=3147,

        depth=2,

        use_best_model=True,

        early_stopping_rounds=200,

        eval_metric='AUC',

    )

    return model
model = getModel()

model.fit(train_X, train_Y,

          eval_set=(val_X, val_Y),

          verbose=False)

from sklearn.metrics import roc_auc_score



p = model.predict_proba(val_X)[:, 1]

print('validation ROC AUC:', roc_auc_score(val_Y, p))
test_p = model.predict_proba(test_df[f_cols].values)[:, 1]

pd.DataFrame({'ID_code': test_df['ID_code'], 'target': test_p}).to_csv('catboost_naive_submission.csv', index=False)
from IPython.display import FileLink

FileLink('catboost_naive_submission.csv')
def shuffleDataInCols(data):

    for i in range(data.shape[1]):

        col = data[:, i].copy()

        np.random.shuffle(col)

        data[:, i] = col

    return data



mask_zero = val_Y == 0

mask_one = val_Y == 0



val_X_shuffled = val_X.copy()

val_X_shuffled[mask_zero] = shuffleDataInCols(val_X_shuffled[mask_zero])

val_X_shuffled[mask_one] = shuffleDataInCols(val_X_shuffled[mask_one])



p_shuffled = model.predict_proba(val_X_shuffled)[:, 1]

print('validation ROC AUC on a shuffled data:', roc_auc_score(val_Y, p_shuffled))
def getCountEncoding(X):

    hist_df = pd.DataFrame()

    

    for var in f_cols:

        var_stats = X[var].value_counts()

        hist_df[var] = pd.Series(X[var]).map(var_stats)

    return hist_df

        
train_ce = getCountEncoding(train_df)

test_ce = getCountEncoding(test_df)

train_ce.head(10)
X = np.concatenate([train_df[f_cols].values, train_ce[f_cols].values], axis=1)

Y = train_df['target'].values

print(X.shape)



train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
model = getModel()

model.fit(train_X, train_Y,

          eval_set=(val_X, val_Y),

          verbose=False)
p = model.predict_proba(val_X)[:, 1]

print('validation ROC AUC:', roc_auc_score(val_Y, p))
test_X = np.concatenate([test_df[f_cols].values, test_ce[f_cols].values], axis=1)

test_p = model.predict_proba(test_X)[:, 1]

pd.DataFrame({'ID_code': test_df['ID_code'], 'target': test_p}).to_csv('catboost_count_encoding_submission.csv', index=False)



from IPython.display import FileLink

FileLink('catboost_count_encoding_submission.csv')
train_unique_cnt = np.sum(train_ce.values==1, axis=1)

test_unique_cnt = np.sum(test_ce.values==1, axis=1)



f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,10))

_ = ax[0].hist(train_unique_cnt, bins=100)

_ = ax[1].hist(test_unique_cnt, bins=100)
mask_real = test_unique_cnt > 0

mask_fake = test_unique_cnt == 0



print('Real samples:', np.sum(mask_real))

print('Fake samples:', np.sum(mask_fake))
test_real_df = test_df[mask_real]

train_test_features = pd.concat([train_df[f_cols], test_real_df[f_cols]], axis=0, ignore_index=True)



train_test_ce = getCountEncoding(train_test_features)

print(train_test_features.shape)
uniques_train_test = np.sum(train_test_ce.values==1, axis=1)

_ = plt.hist(uniques_train_test, bins=100)
X = np.concatenate([train_df[f_cols].values, train_test_ce[f_cols].values[0:train_df.shape[0]]], axis=1)

Y = train_df['target'].values

print(X.shape)



train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
model = getModel()

model.fit(train_X, train_Y,

          eval_set=(val_X, val_Y),

          verbose=False)
p = model.predict_proba(val_X)[:, 1]

print('validation ROC AUC:', roc_auc_score(val_Y, p))
real_X = np.concatenate([test_real_df[f_cols].values, train_test_ce[f_cols].values[train_df.shape[0]:, :] ], axis=1)

test_p = model.predict_proba(real_X)[:, 1]



res = np.zeros(test_df.shape[0], dtype=np.float32)

res[mask_real] = test_p

pd.DataFrame({'ID_code': test_df['ID_code'], 'target': res}).to_csv('catboost_count_encoding_with_real_test_submission.csv', index=False)



from IPython.display import FileLink

FileLink('catboost_count_encoding_with_real_test_submission.csv')