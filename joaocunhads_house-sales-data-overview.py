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
df_housesales = pd.read_csv('../input/kc_house_data.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['housesales']
pd_data    = [len(df_housesales)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_housesales.shape
df_housesales.head()
df_housesales.describe()
df_housesales.hist(column='bedrooms', bins=20)
df_housesales.hist(column='floors')