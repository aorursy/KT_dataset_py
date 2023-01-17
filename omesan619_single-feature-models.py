import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

import gc

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/data-without-drift/train_clean.csv')

test_df  = pd.read_csv('../input/data-without-drift/test_clean.csv')
def apply_group():

    for i in range(20):

        train_df.loc[((train_df.time) > i * 50) & (train_df.time <= (i+1) * 50), 'batch'] = i + 1

        test_df.loc[i*100_000:(i+1)*100_000, 'batch'] = i

    

    batch_group = [(1,0), (2,0), (3,1), (4,2), (5,4), (6,3), (7,1), (8,2), (9,3), (10,4),

                   (11,4),(12,3),(13,2),(14,1),(15,3),(16,4),(17,2),(18,1),(19,0),(20,0)]

    for batch_i, group_i in batch_group:

        train_df.loc[train_df.batch == batch_i, 'group'] = group_i

    

    batch_group = [(1,0), (2,2), (3,3), (4,0), (5,1), (6,4), (7,3), (8,4), (9,0), (10,2),

                         (21,0),(22,0),(23,0),(24,0),(25,0),(26,0),(27,0),(28,0),(29,0),(30,0),

                         (31,2),(32,0),(33,4),(34,3),(35,4),(36,1),(37,0),(38,3),(39,2),(40,0)]

    batch_group.extend([(i, 0) for i in range(11,21)])

    for batch_i, group_i in batch_group:

        test_df.loc[test_df.batch == batch_i, 'group'] = group_i
apply_group()
plt.figure(figsize=(20,5))

for _ in train_df.group.unique():

    plt.plot(train_df[train_df.group == _].set_index('time').signal[::1000], '.')
plt.figure(figsize=(20,5))

for _ in train_df.group.unique():

    plt.plot(test_df[test_df.group == _].set_index('time').signal[::200], '.')
from sklearn.linear_model import LinearRegression, LogisticRegression
diff = {}; alpha = {}; beta = {}

for _ in train_df.group.unique():

    temp = train_df[train_df.group == _]

    beta[_] = np.cov(temp.signal, temp.open_channels)[0,1] / np.var(temp.signal.astype(np.float64))

    alpha[_] = np.mean(temp.open_channels) - (beta[_] * np.mean(temp.signal.astype(np.float64)))

    diff[_] = temp.open_channels - (beta[_] * temp.signal + alpha[_])
alpha, beta
def ls(group_i, start, stop):

    train_df[train_df.group == group_i].sample(1000).plot.scatter(x='signal', y='open_channels', figsize=(18,6))

    

    ols = lambda x, i: beta[i] * x + alpha[i]

    plt.plot(np.linspace(start, stop), np.linspace(ols(start, group_i), ols(stop, group_i)), label='ols')

    

    lr = LogisticRegression(multi_class='multinomial')

    lr.fit(

        train_df[train_df.group == group_i].signal.values.reshape(-1,1),

        y=train_df[train_df.group == group_i].open_channels.values.reshape(-1,1)

    )

    plt.plot(np.linspace(start, stop), lr.predict(np.linspace(start, stop).reshape(-1,1)), label='multinomial')

    

    lr = LogisticRegression(multi_class='ovr')

    lr.fit(

        train_df[train_df.group == group_i].signal.values.reshape(-1,1),

        y=train_df[train_df.group == group_i].open_channels.values.reshape(-1,1)

    )

    plt.plot(np.linspace(start, stop), lr.predict(np.linspace(start, stop).reshape(-1,1)), label='ovr')



    plt.legend(loc='upper left', fontsize=10)

    plt.show()
ls(0, -3.5, -1)
ls(1, -4, -0.5)
ls(2, -4, 3)
ls(3, -4, 4)
ls(4, -4, 8)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=8982)



from sklearn import metrics
def ls_cv(model, cv, group_i, feature_cols):

    

    # Drop noise

    if group_i == 2:

        temp = train_df.drop(train_df.loc[3642932:3822753].index)[train_df.group == group_i]

    else:

        temp = train_df[train_df.group == group_i]

    train_df_idx = temp.index

        

    # Define empty oof / Fill missing values with mean if present

    oof = np.zeros(temp.shape[0])

    temp = temp.fillna(temp.astype(np.float32).mean())

    

    # Cross-validate

    models = []

    for train_idx, valid_idx in cv.split(temp, temp.open_channels):

        

        if issubclass(model, LinearRegression):

            lr = model()

        elif issubclass(model, LogisticRegression):

            lr = model(multi_class='multinomial')

        

        lr.fit(temp[feature_cols].iloc[train_idx].values,

               temp.open_channels.iloc[train_idx].values.reshape(-1,1))

        oof[valid_idx] = lr.predict(temp[feature_cols].iloc[valid_idx].values).flatten()

        models.append(lr)

    

    # Predict OOF

    try:

        valid_f1 = metrics.f1_score(temp.open_channels, oof, average='macro')

    except ValueError:

        oof = np.round(np.clip(oof, 0, 10)).astype(np.int8)

        valid_f1 = metrics.f1_score(temp.open_channels, oof, average='macro')

    

    print(f'valid_f1 of group {int(group_i)}: {valid_f1}')

    

    # Predict on test set

    temp = test_df[test_df.group == group_i]

    temp = temp.fillna(temp.astype(np.float64).mean())

    test_df_idx = temp.index

    

    y_test = np.zeros(temp.shape[0])

    for lr in models:

        y_test += lr.predict(temp[feature_cols].values).flatten()

    y_test /= len(models)

    

    del temp, models

    gc.collect()

    

    return y_test, oof, train_df_idx, test_df_idx
score_linear = {}

for i in train_df.group.unique():

    _ = ls_cv(LinearRegression, kf, i, ['signal'])

    test_df.loc[_[3], 'linear'] = _[0]

    train_df.loc[_[2], 'linear'] = _[1]
score_logistic = {}

for i in train_df.group.unique():

    _ = ls_cv(LogisticRegression, kf, i, ['signal'])

    test_df.loc[_[3], 'logistic'] = _[0]

    train_df.loc[_[2], 'logistic'] = _[1]
def blend_thresholder(oofs, y_tests, col_1, col_2, blend_name):

    

    best = {i: 0 for i in range(5)}

    threshold = {}

    start = 0.0

    end = 1.0

    

    def _print(improved: bool):

        if improved:

            if _ == end:

                print('!')

            else:

                print('!', end='')

        else:

            if _ == end:

                print('.')

            else:

                print('.', end='')

    

    for i in range(5):

        print(f'[Thresholder] ({i})', end=' ')

        

        for _ in np.linspace(start, end, 50):

            temp = _ * oofs[col_1] + (1 - _) * oofs[col_2]

            mask = oofs.group == i

            one = oofs.open_channels.drop(oofs.open_channels.loc[3642932:3822753].index)[mask]

            two = temp.drop(oofs.loc[3642932:3822753].index)[mask]

            score = metrics.f1_score(one,

                                     np.round(np.clip(two, 0, 10)).astype(np.int8),

                                     average='macro')

            if score > best[i]:

                _print(True)

                best[i] = score

                threshold[i] = _

            else:

                _print(False)

                

        oofs.loc[mask, blend_name] = threshold[i] * oofs[mask][col_1] + (1 - threshold[i]) * oofs[mask][col_2]

        one = oofs.open_channels.drop(oofs.loc[3642932:3822753].index)[mask]

        two = oofs[blend_name].drop(oofs.loc[3642932:3822753].index)[mask]

        

        temp = metrics.f1_score(one, np.round(np.clip(two, 0, 10)), average='macro')

        assert best[i] == temp

        

        mask = oofs.group == i

        temp = threshold[i] * y_tests[mask][col_1] + (1 - threshold[i]) * y_tests[mask][col_2]

        y_tests.loc[mask, blend_name] = temp

    

    del one, two, temp; gc.collect()

    print()

    print('best_threshold -', threshold)

    print('overall_score -', metrics.f1_score(oofs.open_channels.drop(train_df.loc[3642932:3822753].index),

                                              np.round(np.clip(oofs[blend_name].drop(train_df.loc[3642932:3822753].index), 0, 10)), average='macro'))
blend_thresholder(train_df, test_df, 'linear', 'logistic', 'ls')
train_df.ls[::2000].plot(figsize=(20,5))
test_df.ls[::900].plot(figsize=(20,5))
sample_submission = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

sample_submission['open_channels'] = test_df.ls.astype(np.int8)

sample_submission.to_csv('submission.csv', index=False)