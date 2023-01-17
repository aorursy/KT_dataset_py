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
submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

my_sub = pd.read_csv('/kaggle/input/mysub-011/submission_small.csv')



submission['accuracy_group'] = 1



for i, row in my_sub.iterrows():

    submission.loc[submission['installation_id'] == row['installation_id'], 'accuracy_group'] = row['accuracy_group']



submission.to_csv('submission.csv', index=False)
submission.head()
submission.tail()