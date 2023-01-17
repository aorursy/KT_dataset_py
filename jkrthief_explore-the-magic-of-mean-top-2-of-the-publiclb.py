import numpy as np

import pandas as pd

import os

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



public_ids = list(submission['sig_id'].values)



submission.shape
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test_ids = list(test['sig_id'].values)



private_ids = list(set(test_ids)-set(public_ids))



len(private_ids)
import glob



def make_filename(idx):

    return glob.glob('../input/moa-1082020/' + str(idx) + '.csv')[0]



def read_predictions(idx):

    temp = pd.read_csv(make_filename(idx))

    return temp





# predictions = [read_predictions(idx) for idx in range(1,18)]

# len(predictions)
predictions = []

predictions.append(pd.read_csv('../input/blendings/submissionfeats.csv'))

predictions.append(pd.read_csv('../input/blendings/submissionorj.csv'))
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
submission = pd.DataFrame(index = public_ids + private_ids, columns=target_columns)

submission.index.name = 'sig_id'



submission[:] = 0



submission.loc[y_pred.sig_id,:] = y_pred[target_columns].values



submission.loc[test[test.cp_type == 'ctl_vehicle'].sig_id] = 0



submission.to_csv('submission.csv', index=True)
