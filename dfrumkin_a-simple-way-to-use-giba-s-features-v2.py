import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
FEATURES = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', 
            '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 
            'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
            '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 
            'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
            '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
            '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', 
            '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
    
def get_pred(data, lag=2):
    d1 = data[FEATURES[:-lag]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[FEATURES[lag:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[FEATURES[lag - 2]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)
def get_all_pred(data, max_lag):
    target = pd.Series(index=data.index, data=np.zeros(data.shape[0]))
    for lag in range(2, max_lag + 1):
        pred = get_pred(data, lag)
        mask = (target == 0) & (pred != 0)
        target[mask] = pred[mask]
    return target
train = pd.read_csv('../input/train.csv')
for max_lag in range(2, 33):
    pred_train = get_all_pred(train, max_lag)
    have_data = pred_train != 0
    print(f'Max lag {max_lag}: Score = {sqrt(mean_squared_error(np.log1p(train.target[have_data]), np.log1p(pred_train[have_data])))} on {have_data.sum()} out of {train.shape[0]} training samples')
test = pd.read_csv('../input/test.csv')
pred_test = get_all_pred(test, 25)
have_data = pred_test != 0
print(f'Have predictions for {have_data.sum()} out of {test.shape[0]} test samples')
# TODO: USE YOUR OLD SUBMISSION!!!
sub = pd.read_csv('../input/sample_submission.csv')
sub.loc[have_data, 'target'] = pred_test[have_data]
sub.to_csv(f'new_submission.csv', index=False)
