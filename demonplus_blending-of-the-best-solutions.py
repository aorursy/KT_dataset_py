# import libraries

import pandas as pd



# load in the submissions

sub1 = pd.read_csv('../input/overfitting-dataset/submission1.csv')

sub2 = pd.read_csv('../input/overfitting-dataset/submission2.csv')

sub3 = pd.read_csv('../input/overfitting-dataset/submission3.csv')

sub4 = pd.read_csv('../input/overfitting-dataset/submission4.csv')
# create blend of submissions

submission = pd.DataFrame()

submission['id'] = sub1['id']

submission['target'] = 0.25*sub1['target']+0.25*sub2['target']+0.25*sub3['target']+0.25*sub4['target']
submission.to_csv('submission.csv', index=False)