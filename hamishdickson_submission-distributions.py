# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import f1_score
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.target.value_counts(normalize=True)
len(df_train)
f1_score(df_train.target.values, np.ones(len(df_train)))
df_test['prediction'] = 1
df_test = df_test[['qid', 'prediction']]

df_test.to_csv('submission.csv', index=False)
positive_ratio = 0.113 / (2 - 0.113)

print(positive_ratio)
0.11653 / (2 - 0.11653)
