import numpy as np
import pandas as pd
data1 = pd.read_csv('../input/siimmy-submissions/submission (3).csv')

data2 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_mean.csv')

data3 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_median.csv')

data4 = pd.read_csv('../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled.csv')

data5 = pd.read_csv('../input/new-basline-np-log2-ensemble-top-10/submission.csv')

data6 = pd.read_csv('../input/minmax-highest-public-lb-9619/submission.csv')

submission = data1.copy()
ncol = data1.shape[1]

data2['target'] = data4.iloc[:, 1:ncol].mean(axis=1)
data3['target'] = data4.iloc[:, 1:ncol].median(axis=1)

submission['target'] =  1/6 * data1['target'] + 1/6 * data2['target'] + 1/6 * data3['target'] + 1/6 *data4['target']+ 1/6 *data5['target']+ 1/6 *data6['target']
submission.to_csv('submission.csv', index=False, float_format='%.6f')
s1= pd.read_csv('../input/siimmy-submissions/submission (6).csv')
s2= pd.read_csv('../input/siimmy-submissions/submission (7).csv')
s3= pd.read_csv('../input/siimmy-submissions/b3submission.csv')

submission= s1.copy()

submission['target'] = 1/3 * s1['target'] + 1/3 * s2['target'] + 1/3 * s3['target'] 

submission.to_csv('b3submission.csv', index=False, float_format='%.6f')
