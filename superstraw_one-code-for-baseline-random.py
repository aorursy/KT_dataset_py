# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
sample_submission.columns
# [3]*sample_submission.shape[0]
import random

import numpy as np
w=np.array([1,3,7,9])

w=w/sum(w)

print(w)

w_l = [int(sample_submission.shape[0]*i) for i in w]

w_l[3]=sample_submission.shape[0]-sum(w_l[:-1])

r = [0]*(w_l[0])+[1]*(w_l[1])+[2]*(w_l[2])+[3]*(w_l[3])

random.shuffle(r)

w_l
r
sample_submission['accuracy_group']=r
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)