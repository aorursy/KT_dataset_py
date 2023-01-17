# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime
from pandas import DataFrame
df_original = DataFrame(
            {
                "Name": "Maria Maria Maria Maria Jane Carlos".split(),
                "Sample": [25, 9, 4, 3, 2, 8],
                "Date": [
                    datetime(2019, 9, 1, 13, 0),
                    datetime(2019, 9, 1, 13, 5),
                    datetime(2019, 10, 1, 20, 0),
                    datetime(2019, 10, 3, 10, 0),
                    datetime(2019, 12, 2, 12, 0),
                    datetime(2019, 9, 2, 14, 0),
                ],
            }
        )
df_original.set_index('Date',inplace=False)
df_original
import pandas as pd
df_original.groupby(pd.Grouper(key = "Date",freq="M")).sum()
df = pd.read_excel("https://github.com/chris1610/pbpython/blob/master/data/sample-salesv3.xlsx?raw=True")
df["date"] = pd.to_datetime(df['date'])
df.head()
import pandas as pd
df.groupby(pd.Grouper(key = "date",freq="M")).sum()
import pandas as pd
df_original.groupby(pd.Grouper(key='Date',freq='5D')).sum()
import pandas as pd 
df_original_5D = df_original.groupby(pd.Grouper(key='Date', freq='5D')).sum()
df_original_5D[df_original_5D['Sample']!=0]
import pandas as pd 
df_5D = df.groupby(pd.Grouper(key='date', freq='5D')).sum()
df_5D[df_5D['quantity'] >= 700]
import pandas as pd 
df_5D = df.groupby(pd.Grouper(key='date', freq='5D')).sum()
df_5D[df_5D['unit price'] >= 1300]
df_original.set_index('Name', inplace=True)
df_original.groupby(pd.Grouper(level= 'Name', axis = 0)).sum()
df.set_index('date').resample('M')["ext price"].sum()
df.set_index('date').groupby('name')["ext price"].resample("M").sum()
df.groupby(['name', pd.Grouper(key='date', freq='M')])['ext price'].sum()
df.groupby(['name', pd.Grouper(key='date', freq='A-DEC')])['ext price'].sum()
df[["ext price", "quantity"]].sum()
df["unit price"].mean()
df[["ext price", "quantity", "unit price"]].agg(['sum', 'mean'])
df.agg({'ext price': ['sum', 'mean'], 'quantity': ['sum', 'mean'], 'unit price': ['mean']})
get_max = lambda x: x.value_counts(dropna=False).index[0]
df.agg({'ext price': ['sum', 'mean'], 'quantity': ['sum', 'mean'], 'unit price': ['mean'], 'sku': [get_max]})
import collections
f = collections.OrderedDict([('ext price', ['sum', 'mean']), ('quantity', ['sum', 'mean']), ('sku', [get_max])])
df.agg(f)
