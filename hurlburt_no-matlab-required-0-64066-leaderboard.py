# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np

import pandas as pd

sdf = pd.read_csv('../input/sample_submission.csv', index_col=0)



p1files = [x for x in sdf.index if x.split('_')[1] == '1']  #patient1 files

p2files = [x for x in sdf.index if x.split('_')[1] == '2']  #patient2 files

p3files = [x for x in sdf.index if x.split('_')[1] == '3']  #patient3 files



sdf['Class'] = .5

sdf.loc[p1files,'Class'] = .6

sdf.loc[p2files,'Class'] = .4



sdf.to_csv('submission_roc_test.csv')

sdf.head()
sdf.tail()
sdf.iloc[300:305]