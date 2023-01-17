import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



import os

print(os.listdir("../input"))
sub1 = pd.read_csv('../input/stacking/stack_median.csv')

sub2 = pd.read_csv('../input/stacking-higher-and-higher/stack_median.csv')

sub3 = pd.read_csv('../input/ensemble/submission_p2_1.csv')

sub4 = pd.read_csv('../input/safe-box/blend02.csv')

sub5 = pd.read_csv('../input/stackers-blend-top-4/All_Blends.csv')

sample = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
sample['isFraud'] = (0.35*sub1['isFraud'] + 0.30*sub2['isFraud'] + 0.25*sub3['isFraud'] + 0.15*sub4['isFraud'] + 0.15*sub5['isFraud'])

sample.to_csv('submission.csv', index=False)
sns.distplot(sample['isFraud']);