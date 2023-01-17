# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from datetime import datetime
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/payments/data.csv',  names = ["id", "created_at", "user_id", "amount"])
df.dtypes
df['DATE']= df['created_at'].str.split(' ', 1).str[0]
day = []

for i in df.DATE:

    different = datetime.strptime(i, "%Y-%m-%d")-datetime.strptime('2017-11-07', "%Y-%m-%d")
    day.append(different.days)

df['Day'] = day
df.head()
df.sort_values(by=['Day'])
cs = df.groupby('Day').mean()
cs['Average revenue per customer'] = cs.amount
cs = cs['Average revenue per customer']
cs.head()
cs.to_csv('gust_challenge.csv', sep=',')
