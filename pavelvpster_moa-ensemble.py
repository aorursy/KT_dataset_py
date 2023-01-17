import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



public_ids = list(submission['sig_id'].values)



submission.shape
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test_ids = list(test['sig_id'].values)



private_ids = list(set(test_ids)-set(public_ids))



len(private_ids)
kernels = pd.read_csv('/kaggle/input/moa-public-kernels/kernels.csv', index_col='id')
kernels.head(10)
import glob



def make_filename(idx):

    return glob.glob('/kaggle/input/moa-public-kernels/' + str(idx) + '__submission.csv')[0]



def read_predictions(idx):

    temp = pd.read_csv(make_filename(idx))

    return temp





predictions = [read_predictions(idx) for idx in kernels.index]

len(predictions)
target_columns = list(submission.columns)

target_columns.remove('sig_id')
y_pred = pd.DataFrame()

y_pred['sig_id'] = predictions[0]['sig_id']



for column in target_columns:

    column_data = []

    for i in range(len(predictions)):

        column_data.append(predictions[i][column])

    y_pred[column] = np.mean(column_data, axis=0)



y_pred.shape
# (1 - score) because lesser metric is better

scores = 1. - kernels['score']



sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
weighted_y_pred = pd.DataFrame()

weighted_y_pred['sig_id'] = predictions[0]['sig_id']



for column in target_columns:

    column_data = []

    for i in range(len(predictions)):

        column_data.append(predictions[i][column] * weights[i])

    weighted_y_pred[column] = np.sum(column_data, axis=0)



weighted_y_pred.shape
y_pred = weighted_y_pred
submission = pd.DataFrame(index = public_ids + private_ids, columns=target_columns)

submission.index.name = 'sig_id'



submission[:] = 0



submission.loc[y_pred.sig_id,:] = y_pred[target_columns].values



submission.loc[test[test.cp_type == 'ctl_vehicle'].sig_id] = 0



submission.to_csv('submission.csv', index=True)



submission.head().T