import numpy as np

import pandas as pd
data1 = pd.read_csv('../input/minmax-ensemble-0-9526-lb/submission.csv')



data2 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_mean.csv')



data3 = pd.read_csv('../input/stacking-ensemble-on-my-submissions/submission_median.csv')



data4 = pd.read_csv('../input/analysis-of-melanoma-metadata-and-effnet-ensemble/ensembled.csv')



data5 = pd.read_csv('../input/new-basline-np-log2-ensemble-top-10/submission.csv')



submission = data1.copy()
submission['target'] = 2/6 * data1['target'] + 1/6 * data2['target'] + 1/6 * data3['target'] + 1/6 * data4['target'] + 1/6 * data5['target']
submission.to_csv('submission.csv', index=False, float_format='%.6f')