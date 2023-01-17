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
import numpy as np

import pandas as pd
from datetime import datetime

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv('/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv', parse_dates=['event_time'], date_parser=dateparse)
unique_shop_ids = df["shopid"].unique()
def check_period(hour_mask, shop_orders, df):

  shop_orders_within_hour = shop_orders[hour_mask]



  order_within_hour_count = len(shop_orders_within_hour)



  unique_buyers = shop_orders_within_hour['userid'].unique()

  unique_buyers_count = len(shop_orders_within_hour['userid'].unique())



  ratio = order_within_hour_count / unique_buyers_count



  if (ratio < 3.0):

    return



  for hourly_order_id in range(len(shop_orders_within_hour)):

    order = shop_orders_within_hour.iloc[hourly_order_id]

    df.loc[df['orderid'] == order['orderid'], 'concentrate'] = True



result = {}

df['concentrate'] = False

for shop_id in unique_shop_ids:

  result[shop_id] = {}



  shop_orders = df.loc[df['shopid'] == shop_id].sort_values('event_time', ascending=True)



  for order_id in range(len(shop_orders)):

    end_time = shop_orders.iloc[order_id]['event_time']

    start_time = end_time - pd.DateOffset(seconds=(60*60 - 1))

    hour_mask = (shop_orders['event_time'] >= start_time) & (shop_orders['event_time'] <= end_time)

    check_period(hour_mask, shop_orders, df)



    start_time = shop_orders.iloc[order_id]['event_time']

    end_time = end_time + pd.DateOffset(seconds=(60*60 - 1))

    hour_mask = (shop_orders['event_time'] >= start_time) & (shop_orders['event_time'] <= end_time)

    check_period(hour_mask, shop_orders, df)
for shop_id in unique_shop_ids:

  shop_orders = df.loc[df['shopid'] == shop_id].sort_values('event_time', ascending=True)



  shop_user_order_counts = shop_orders[shop_orders['concentrate'] == True]['userid'].value_counts()

  result[shop_id] = shop_user_order_counts
final_result = []

for shop_id in result:

  if (len(result[shop_id]) == 0):

    final_result.append([shop_id, '0'])

    continue

  

  highest_count = result[shop_id].iloc[0]

  users = []

  for user in result[shop_id].index:

    if (highest_count == result[shop_id][user]):

      users.append(user)

  

  

  unique_users = [str(i) for i in users] 

  value = '&'.join(unique_users)



  final_result.append([shop_id, value])
csv = pd.DataFrame(final_result, columns={"shopid", "userid"}).sort_values('shopid', ascending=True)
csv.to_csv('filename.csv', index = False)