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
df_gbcomments = pd.read_csv('../input/GBcomments.csv', error_bad_lines=False)
df_gbvideos = pd.read_csv('../input/GBvideos.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['comments', 'videos']
pd_data    = [len(df_gbcomments), len(df_gbvideos)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_gbcomments.head()
df_gbcomments.describe()
df_gbvideos.head()
df_gbvideos.describe()
df_uscomments = pd.read_csv('../input/UScomments.csv', error_bad_lines=False)
df_usvideos = pd.read_csv('../input/USvideos.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['comments', 'videos']
pd_data_US    = [len(df_uscomments), len(df_usvideos)]

pd.DataFrame(pd_data_US, index = pd_index, columns = pd_columns)
df_uscomments.head()
df_uscomments.describe()
df_usvideos.head()
df_usvideos.describe()