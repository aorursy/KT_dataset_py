import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
averages = train_labels.groupby('title')['accuracy_group'].agg(['median','count'])
averages
ans = {'Bird Measurer (Assessment)':1,

       'Cart Balancer (Assessment)': 3,

       'Cauldron Filler (Assessment)':3,

       'Chest Sorter (Assessment)': 0,

       'Mushroom Sorter (Assessment)':3

      }
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
submission['accuracy_group'] = test.groupby('installation_id').last().title.map(ans).reset_index(drop=True)
submission.to_csv('submission.csv', index=None)