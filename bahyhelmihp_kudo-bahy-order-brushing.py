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
import pandas as pd



## Load dataset

df = pd.read_csv("/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv") 
df.head()
## Quick analysis

df.describe()
## Number of unique seller

df['shopid'].nunique()
## Number of unique seller

df['userid'].nunique()
## Average trx each shop has

df.groupby('shopid')['orderid'].count().mean()
## Minimum time 

df[df['event_time'] == df['event_time'].min()]
## Set event_time as index 

df_time = df.set_index(pd.DatetimeIndex(df['event_time'])).drop('event_time', axis=1)

df_time = df_time.sort_index()



## Groupby shopid, userid, and event_time 60 min intervals, count order(s)

grouped_orders = df_time.groupby(['shopid', 'userid', pd.Grouper(freq='60min', label='left', base=0)]).count()
## Find shopid that possibly do order brushing, given a threshold

possible_brush = grouped_orders[grouped_orders.orderid > 2]



userids = []

possible_brush.reset_index().groupby('shopid')['userid'].apply(lambda x: userids.append(x.values))

len(userids)
possible_brush.reset_index().shopid.nunique()
def concat_and(arr):

    res = '&'.join(str(x) for x in arr)

    return res



concat_userids = []

for element in userids:

    concat_userids.append(concat_and(element))



res_df = pd.DataFrame({"shopid": possible_brush.reset_index()['shopid'].unique(), "userid": concat_userids})

res_df.head()
## Final answer

default_df = pd.DataFrame({'shopid': df['shopid'].unique(), 'userid': 0})

default_df.head()
res_df.head()
res_df.shape
## Export Result

sol_df = pd.concat([default_df[~default_df.shopid.isin(res_df.shopid)], res_df])

sol_df.to_csv("solution_3.csv", index=False)
## Check result

pd.read_csv("solution_3.csv")