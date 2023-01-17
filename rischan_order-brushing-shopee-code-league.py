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
# Read the data 

df = pd.read_csv('/kaggle/input/orderbrushing/order_brush_order.csv')
df.head()
df.describe()
# Convert string to date time type Python

df["event_time"] = pd.to_datetime(df['event_time'])
#Get all orders with group by userid and shopid

df = df.set_index(pd.DatetimeIndex(df['event_time'])).drop('event_time', axis=1).sort_index()

orders = df.groupby(['shopid', 'userid', pd.Grouper(freq='H', label='left', base=0)]).count()
orders
brush_order = orders[orders.orderid >=3]

brush_order
listuserid = []

brush_order.reset_index().groupby('shopid')['userid'].apply(lambda x: listuserid.append(x.values))
#Check list userid

listuserid
#Drop duplicate shopid

brush_order.reset_index().drop_duplicates(subset = ["shopid"])
#Concat userid with &

def concat_userid(data):

    result = '&'.join(str(x) for x in data)

    return result



bulk_userid = []

for i in listuserid:

    bulk_userid.append(concat_userid(i))
bulk_userid
#DF order brushing

df_brush = pd.DataFrame({"shopid": brush_order.reset_index()['shopid'].unique(), "userid": bulk_userid})

df_brush.head()
#DF no order brushing

df0 = pd.DataFrame({'shopid': df['shopid'].unique(), 'userid': 0})
# Export result as csv

res_df = pd.concat([df0[~df0.shopid.isin(df_brush.shopid)], df_brush])

res_df.to_csv("submission.csv", index=False)