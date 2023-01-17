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
df_bike = pd.read_csv('../input/data.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['bikesharing']
pd_data    = [len(df_bike)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_bike.shape
df_bike.head()
df_bike.describe()
from collections import Counter
gender_counter = Counter(df_bike['gender'])
df = pd.DataFrame.from_dict(gender_counter, orient='index')
df.plot(kind='bar')
usertype_counter = Counter(df_bike['usertype'])
df = pd.DataFrame.from_dict(usertype_counter, orient='index')
df.plot(kind='bar')