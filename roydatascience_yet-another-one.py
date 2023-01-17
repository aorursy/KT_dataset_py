import numpy as np 

import pandas as pd 

import seaborn as sns



import os

print(os.listdir("../input"))
sub1 = pd.read_csv('../input/mysubmissions/submission(-1.581).csv')

sample = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')

sub2 = pd.read_csv('../input/mysubmissions/submission(-1.587).csv')
sub1['scalar_coupling_constant'].describe()
sub2['scalar_coupling_constant'].describe()
sample['scalar_coupling_constant'] = sub2['scalar_coupling_constant'] 

sample.to_csv('stackers_blend.csv', index=False)
sns.distplot(sample['scalar_coupling_constant'])