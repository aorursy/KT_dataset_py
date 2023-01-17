# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



!ls ../input
!ls ../
TRAIN = "../input/train.csv"

TEST = "../input/test.csv"



df_train = pd.read_csv(TRAIN)

df_test = pd.read_csv(TEST)
print("The shape of train data: ", df_train.shape)

df_train.head()
df_train["is_duplicate"].value_counts()
sns.distplot(df_train["is_duplicate"])
from sklearn.metrics import log_loss



p = df_train['is_duplicate'].mean() # Our predicted probability

print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))



df_test = pd.read_csv('../input/test.csv')

sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})

sub.to_csv('../output/naive_submission.csv', index=False)

sub.head()