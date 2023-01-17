import numpy as np

import pandas as pd

import pickle



with open('../input/prediction//prediction.pkl', 'rb') as f:

    preds = pickle.load(f)



sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')

if sub.shape[0] != 131073:

    sub['target'] = preds

sub.to_csv('submission.csv',index=False)