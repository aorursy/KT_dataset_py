import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# write fake submission file 

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = 0

sub.to_csv('submission.csv',index=False)
test = pd.read_csv('../input/test.csv')
if len(test) < 150000:

    [].shape

    

# your fancy code goes here

sleep(100000000000)





# make sure to rewrite your submission file here