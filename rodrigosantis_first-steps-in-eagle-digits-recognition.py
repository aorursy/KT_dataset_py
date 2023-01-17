# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train.head(3)
df_test.head(3)
y_train = df_train['label'].values

X_train = df_train.iloc[:, 1:785].values

y_test = df_test.values