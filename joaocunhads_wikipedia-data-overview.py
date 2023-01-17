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
df_train_1 = pd.read_csv('../input/train_1.csv', error_bad_lines=False)
df_train_2 = pd.read_csv('../input/train_2.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['train_1', 'train_2']
pd_data    = [len(df_train_1), len(df_train_2)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_train_1.head()
df_train_1.shape
df_train_1.describe()