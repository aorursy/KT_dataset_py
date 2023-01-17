import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import os, time, gc

from tqdm import tqdm_notebook



import librosa

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor



os.listdir('../input')
%%time



train = pd.read_csv(

    '../input/train.csv', 

    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print('loaded:', train.shape)
train.head()
data = train['acoustic_data'].values[::200]

ttf = train['time_to_failure'].values[::200]



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

ax.plot(data, label='acoustic_data')

ax.plot(ttf * 100, label='time_to_feailure')

plt.legend()

plt.show()



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

ax.plot(data[:200000], label='acoustic_data')

ax.plot(ttf[:200000] * 100, label='time_to_feailure')

plt.legend()

plt.show()



del data, ttf

gc.collect()
def add(row, key, value, mode=False):

    if mode:

        row.append(key)

    else:

        row.append(value)



def agg(row, d, prefix='', mode=False):

    add(row, 'mean{}'.format(prefix), d.mean(), mode)

    add(row, 'std{}'.format(prefix), d.std(), mode)

    add(row, 'kurt{}'.format(prefix), d.kurt(), mode)

    add(row, 'range{}'.format(prefix), d.max() - d.min(), mode)



    add(row, 'q0.01{}'.format(prefix), np.quantile(d, 0.01), mode)

    add(row, 'q0.05{}'.format(prefix), np.quantile(d, 0.05), mode)    

    add(row, 'q0.95{}'.format(prefix), np.quantile(d, 0.95), mode)

    add(row, 'q0.99{}'.format(prefix), np.quantile(d, 0.99), mode)

    

    add(row, 'iqr{}'.format(prefix), np.subtract(*np.percentile(d, (75, 25))), mode)



def wave2row(d, mode=False):

    row = []



    agg(row, d, mode=mode)

    

    return row



def make_cols():

    cols = wave2row(pd.Series(list(range(150000))), mode=True)

    cols.extend(['seg_id', 'time_to_failure'])

    return cols
plt.figure(figsize=(16, 6))

plt.plot(np.real(np.fft.fft(train[0:150000]['acoustic_data']))[1:75000])

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(librosa.feature.mfcc(

    train[0:150000]['acoustic_data'].values.astype(np.float32), n_mfcc=64)[2:])

plt.show()
def make_train(i, f):

    data = f[i:i + 150000]

    

    row = wave2row(data['acoustic_data'])

    

    add(row, 'seg_id', str(i))

    add(row, 'time_to_failure', data[-1:]['time_to_failure'].values[0])

    

    return row



indexes = [i for i in range(0, len(train), 150000)]

train_rows = []

for i in tqdm_notebook(indexes):

    train_rows.append(make_train(i, train))



train_rows = pd.DataFrame(train_rows, columns=make_cols())

train_rows.to_csv('train.csv', index=False)

print('saved train:', train_rows.shape)



train_rows.head()
def make_test(seg_id):

    data = pd.read_csv('../input/test/' + seg_id + '.csv', dtype={'acoustic_data': np.int16})

    row = wave2row(data['acoustic_data'])

    add(row, 'seg_id', seg_id)

    add(row, 'time_to_failure', 0.0)

    

    return row



submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_rows = []

for seg_id in tqdm_notebook(submission.index):

    test_rows.append(make_test(seg_id))



test_rows = pd.DataFrame(test_rows, columns=make_cols())

test_rows.to_csv('test.csv', index=False)

print('saved test', test_rows.shape)



test_rows.head()
def learn(params):

    local_train = train_rows.copy()

    local_test = test_rows.copy()

    

    y = local_train['time_to_failure']

    x = local_train.drop(['seg_id', 'time_to_failure'], axis=1)

    

    cols = params['cols']

    if len(cols) == 0:

        cols  = list(x.columns)

    print('cols:', len(cols))

    

    cols.sort()



    x = x[cols]

    test_x = local_test.drop(['seg_id', 'time_to_failure'], axis=1)[cols]

    

    x, y = x.values, y.values

    

    fold = KFold(n_splits=5, random_state=params['random_state'], shuffle=True)

    

    val_accs = []

    val_preds = []

    weights = []

    submissions = []

    

    for i, (train_index, val_index) in enumerate(fold.split(x, y)):

        train_x, train_y = x[train_index], y[train_index]

        val_x, val_y = x[val_index], y[val_index]

        

        print('  fold:', i + 1, train_x.shape, val_x.shape)

        

        model = CatBoostRegressor(

            iterations=256, learning_rate=0.015, verbose=32, eval_metric='MAE',

            use_best_model=True, task_type='GPU')

        model.fit(train_x, train_y, eval_set=(val_x, val_y))

        

        val_pred = model.predict(val_x)

        val_pred = np.where(val_pred < 0, 0, val_pred)

        val_acc = mean_absolute_error(val_y, val_pred)

        val_accs.append(val_acc)

        

        val_f = pd.DataFrame()

        val_f['val_index'] = val_index

        val_f['time_to_failure'] = val_pred

        val_preds.append(val_f)



        weight = pd.DataFrame()

        weight['col'] = cols

        weight['weight'] = model.feature_importances_

        weights.append(weight)



        test_pred = model.predict(test_x)

        test_pred = np.where(test_pred < 0, 0, test_pred)

        submission = pd.DataFrame()

        submission['seg_id'] = local_test['seg_id']

        submission['time_to_failure'] = test_pred

        submissions.append(submission)



    print('  ', ['{0:.4f}'.format(acc) for acc in val_accs], '{0:.4f}'.format(np.mean(val_accs)))

    

    weights = pd.concat(weights) if len(weights) > 1 else weights[0]

    val_preds = pd.concat(val_preds) if len(val_preds) > 1 else val_preds[0]

    val_preds.sort_values('val_index', inplace=True)

    val_preds.reset_index(drop=True, inplace=True)

    submissions = pd.concat(submissions) if len(submissions) > 1 else submissions[0]

    

    return submissions, weights, val_accs, val_preds



cols = []

submissions, weights, val_accs, val_preds = learn({ 'cols': cols, 'random_state': 48 })
means = weights.groupby('col', as_index=False).mean().rename(columns={'weight':'mean'})

bars = pd.merge(weights, means, on='col', how='left')

bars.sort_values('mean', ascending=False, inplace=True)



bars.to_csv('weight.csv', index=False)

print('saved weight:', bars.shape)



plt.figure(figsize=(8, 4))

sns.barplot(x='weight', y='col', data=bars)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(16, 4))

ax.plot(train_rows['time_to_failure'])

ax.plot(train_rows['q0.95'])

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(16, 4))

ax.plot(train_rows['time_to_failure'])

ax.plot(val_preds['time_to_failure'])

plt.show()
colormap = plt.cm.RdBu

plt.figure(figsize=(10,8))



sns.heatmap(

    train_rows.drop('seg_id', axis=1).corr(), linewidths=0.1, vmax=1.0, square=True,

    cmap=colormap, linecolor='white', annot=True)

plt.show()
sns.pairplot(train_rows.drop('seg_id', axis=1))

plt.show()
subs = submissions.groupby('seg_id', as_index=False).mean()

subs.to_csv('submission.csv', index=False)

print('saved submission', subs.shape)

subs.head()
sns.distplot(train_rows['time_to_failure'], kde=True, label='train')

sns.distplot(val_preds['time_to_failure'], kde=True, label='val')

sns.distplot(subs['time_to_failure'], kde=True, label='test')

plt.legend()

plt.show()