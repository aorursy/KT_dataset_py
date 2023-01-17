# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_train = pd.read_csv('../input/train.csv')

df_train.head()

df_train[0:9]
duplicate=df_train[df_train.is_duplicate==1]

not_duplicate=df_train[df_train.is_duplicate==0]
duplicate.describe()
print ("Not Duplicates")

(not_duplicate.describe())
not_duplicate.describe()