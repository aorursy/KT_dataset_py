import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
def get_log_pred(data):
    # Need more features!!! Note that if we use
    features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb']
    d1 = data[features[:-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[features[2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[features[0]]
    d2 = d2[d2['pred'] != 0] # Keep?
    d3 = d2[~d2.duplicated(['key'], keep='first')] # Need more features!
    d = d1.merge(d3, how='left', on='key')
    return np.log1p(d.pred).fillna(0)
train = pd.read_csv('../input/train.csv')
log_pred = get_log_pred(train)
have_data = log_pred != 0
print(f'Score = {sqrt(mean_squared_error(np.log1p(train.target[have_data]), log_pred[have_data]))} on {have_data.sum()} out of {train.shape[0]} training samples')
test = pd.read_csv('../input/test.csv')
log_pred = get_log_pred(test)
have_data = log_pred != 0
print(f'Have predictions for {have_data.sum()} out of {test.shape[0]} test samples')
