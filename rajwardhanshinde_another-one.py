import numpy as np 

import pandas as pd 

import seaborn as sns



import os

print(os.listdir("../input"))
sub1 = pd.read_csv('../input/lgb-public-kernels-plus-more-features/sub_lgb_model_individual.csv')

sub2 = pd.read_csv('../input/staking-and-stealing-like-a-molecule/submission.csv')

sample = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')
sub1['scalar_coupling_constant'].describe()
sub2['scalar_coupling_constant'].describe()
sample['scalar_coupling_constant'] = (0.6*sub2['scalar_coupling_constant'] + 0.4*sub1['scalar_coupling_constant'])

sample.to_csv('stackers_blend.csv', index=False)
sns.distplot(sample['scalar_coupling_constant'])