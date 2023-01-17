import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import math

import os
submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

csv1 = pd.read_csv('../input/train-infer-catalyst-pytorch-rnn-baseline/submission.csv')

csv2 = pd.read_csv('../input/neural-covid-vaccine/submission.csv')

csv3 = pd.read_csv('../input/openvaccine-gru-lstm/submission.csv')

csv4 = pd.read_csv('../input/openvaccine-simple-gru-model/submission.csv')
files = [csv1, csv2, csv3, csv4]

total = pd.concat(files, axis=1)
submission.head()
total.head()
col = list(submission.columns)[1:]

print(col)
submission1 = submission

for c in col:

    submission1[c] = total[c].mean(axis=1)

submission1.head()
submission1.to_csv('submission1.csv', index=False)
submission2 = submission

for c in col:

    submission2[c] = total[c].median(axis=1)

submission2.head()
submission2.to_csv('submission2.csv', index=False)
submission3 = submission

for c in col:

    submission3[c] = total[c].mode(axis=1)

submission3.head()
submission3.to_csv('submission3.csv', index=False)
submission4 = submission



cutoff_lo = 0.73

cutoff_hi = 0.33

    

for c in col:

    total[c+'_is_iceberg_max'] = total[c].max(axis=1)

    total[c+'_is_iceberg_min'] = total[c].min(axis=1)

    total[c+'_is_iceberg_mean'] = total[c].mean(axis=1)

    total[c+'_is_iceberg_median'] = total[c].median(axis=1)



    total[c+'_is_iceberg_base'] = csv3[c] #best score

    submission4[c] = np.where(np.all(total[c] > cutoff_lo, axis=1),

                                        total[c+'_is_iceberg_max'],

                                        np.where(np.all(total[c] < cutoff_hi, axis=1),

                                                 total[c+'_is_iceberg_min'],

                                                 total[c+'_is_iceberg_base']))

submission4.head()
submission4.to_csv('submission4.csv', index=False)
submission5 = submission

for c in col:

    submission5[c] = csv3[c] * 0.8 + csv4[c] * 0.1 + csv1[c] * 0.05 + csv2[c] *0.05

submission5.head()
submission5.to_csv('submission5.csv', index=False)