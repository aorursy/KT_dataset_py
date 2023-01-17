import numpy as np

import pandas as pd 



import os

print(os.listdir("../input"))
sub1 = pd.read_csv('../input/blend/submission1236.csv')

sub2 = pd.read_csv('../input/statstack/stack_median.csv')

submission = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')
submission['scalar_coupling_constant'] = (0.4*sub1['scalar_coupling_constant'] + 0.6*sub2['scalar_coupling_constant'])

submission.to_csv('blend_of_blends.csv', index=False)
submission['scalar_coupling_constant']