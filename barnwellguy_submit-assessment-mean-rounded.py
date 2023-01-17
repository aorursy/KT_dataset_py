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
train_labels      = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

test              = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
groupMean = train_labels.groupby('title').accuracy_group.mean()
sample_submission['title'] = test.groupby('installation_id').title.tail(1).values

sample_submission['accuracy_group'] = sample_submission.title.map(groupMean.round().astype(np.int))

sample_submission.drop(columns='title',inplace=True)
assert all(test.groupby('installation_id').installation_id.tail(1).values == sample_submission.installation_id)
sample_submission.to_csv('submission.csv',index=False)