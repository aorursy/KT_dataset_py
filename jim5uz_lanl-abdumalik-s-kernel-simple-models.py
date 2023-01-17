import numpy as np 

import pandas as pd  

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor, Pool

import os

print(os.listdir("../input"))

def new_column(tr):

    add_X = []

    add_X.append(tr.values[0])

    add_X.append(tr.values[-1])

    add_X.append(tr.mean())

    add_X.append(tr.count())

    add_X.append(tr.sum())

    add_X.append(tr.std())

    add_X.append(tr.median())

    add_X.append(tr.min())

    add_X.append(tr.max())

    add_X.append(tr.kurtosis())

    add_X.append(tr.skew())

    return pd.Series(add_X)
#%time

tr_data = pd.read_csv("../input/train.csv", iterator=True, chunksize=600_000)

x_tr_data = pd.DataFrame()

y_tr_data = pd.Series()

for data in tr_data:

    add_tr = new_column(data['acoustic_data'])

    x_tr_data = x_tr_data.append(add_tr, ignore_index=True)

    y_tr_data = y_tr_data.append(pd.Series(data['time_to_failure'].values[-1]))

    

del tr_data

train_pool = Pool(x_tr_data, y_tr_data)

my_model = CatBoostRegressor(iterations=1000, loss_function='MAE', boosting_type='Ordered')

my_model.fit(x_tr_data, y_tr_data, silent=True)

my_model.best_score_
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_data = pd.DataFrame(columns=x_tr_data.columns, dtype=np.float64, index=submission.index)

for seg_id in test_data.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    add_tr = new_column(seg['acoustic_data'])

    test_data.loc[seg_id]= add_tr    

    

submission['time_to_failure'] = my_model.predict(test_data)

submission.to_csv('submission.csv')
print(os.listdir("../input"))








