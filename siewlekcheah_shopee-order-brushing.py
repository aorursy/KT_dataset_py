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
dataset = pd.read_csv('../input/students-order-brushing-1/order_brush_order.csv')
dataset
dataset.dtypes
dataset['event_time'] = pd.to_datetime(dataset['event_time'])
dataset.dtypes
dataset.sort_values(by=['shopid','event_time'], inplace = True)
dataset[0:10]
distinct_shop_id = np.unique(dataset['shopid'].to_numpy())

shops = []
for shop_id in distinct_shop_id:
    shops.append(dataset.loc[dataset['shopid']==shop_id])

def get_suspicious_buyer(shop):
    n_orders = len(shop.index)
    if n_orders < 3:
        return 0
    
    damn_suspicious_buyer_str = []
    
    for i in range(n_orders-2):
        starting_time = shop['event_time'].iloc[i]
        suspicious_buyer = {}
        userid = shop['userid'].iloc[i]
        suspicious_buyer[userid] = 1
        for j in range(i+1, n_orders):
            delta_second = (shop['event_time'].iloc[j] - shop['event_time'].iloc[i]).total_seconds()
            if delta_second > 3601:
                break
            userid = shop['userid'].iloc[j]
            if userid in suspicious_buyer:
                suspicious_buyer[userid] += 1
            else:
                suspicious_buyer[userid] = 1
        n_distinct_users_within_1_hour = len(suspicious_buyer)
        n_orders_within_1_hour = sum(suspicious_buyer.values())
        concentrate_rate = n_orders_within_1_hour / n_distinct_users_within_1_hour
        if concentrate_rate < 3:
            continue
        suspicious_buyer = dict(sorted(suspicious_buyer.items()))
        max_value = max(suspicious_buyer.values())
        l = []
        for key, value in suspicious_buyer.items():
            if value == max_value:
                l.append(str(key))
        damn_suspicious_buyer_str.append('&'.join(l))
        return damn_suspicious_buyer_str[0]
    return 0
shop_ids = []
suspicious_users = []
for shop in shops:
    shop_ids.append(shop['shopid'].iloc[0])
    suspicious_users.append(get_suspicious_buyer(shop))
    
    

output = pd.DataFrame({'shopid': shop_ids,
                       'userid': suspicious_users})
output.to_csv('submission.csv', index=False)
