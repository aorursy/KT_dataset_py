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
d1 = pd.read_csv(os.path.join(dirname, filename), header=0, parse_dates=[3])
d1.dtypes
row, column = d1.shape
print("Row: ", row)
d1.describe().T
d1.head(10)
delta = pd.Timedelta(hours=1)
order_list = set() # store other rows that relevant to a brushing rows
shop_query = dict() # to store query for shopid to quickly access

i = 0
while i < row:
    # if i % 10000 == 0:
    #     print(i)

    r = d1.iloc[i]
    end_time = r.event_time + delta
    previous_time = r.event_time - delta
    
    q = shop_query[r.shopid] if r.shopid in shop_query else d1[d1.shopid == r.shopid]
    shop_query[r.shopid] = q

    q_n = q[q.event_time.between(r.event_time, end_time)]
    q_p = q[q.event_time.between(previous_time, r.event_time)]

    con_rate_p = len(q_p)/q_p.userid.nunique()
    con_rate_n = len(q_n)/q_n.userid.nunique()

    if con_rate_p >= 3:
        # order_brushing.append(r.orderid)
        order_list.update(list(q_p.orderid.unique()))
    if con_rate_n >= 3:
        order_list.update(list(q_n.orderid.unique()))

    i+=1

d2 = d1[d1.orderid.isin(order_list)]
print(d2.shape)
d2_row, d2_column = d2.shape
def find_max(userid_list, user_p_list):
    max_value = max(user_p_list)
    maxs_index = []
    for i, value in enumerate(user_p_list):
        if value == max_value:
            maxs_index.append(i)
    max_user = [ userid_list[i] for i in maxs_index]
    return max_value, set(max_user)
shop_list = dict()

def add_to_list(shopid, max_tuples):
    if shopid not in shop_list:
        shop_list[shopid] = max_tuples
        return
    
    max_value, max_user = shop_list[shopid]
    max_v, max_u = max_tuples
    if max_v > max_value:
        shop_list[shopid] = max_tuples
    elif max_v == max_value:
        shop_list[shopid] = (max_value, max_user.union(max_u))

d2_uniq_shop = d2.shopid.unique()
for shopid in d2_uniq_shop:
    q = d2[d2.shopid == shopid]

    user_list = q.userid.unique()
    sum_order = len(q)
    user_proportion = []
    for userid in user_list:
        user_proportion.append(len(q[q.userid == userid])/sum_order)

    max_value, max_user = find_max(user_list,user_proportion)
    add_to_list(shopid, (max_value, max_user))
unique_shopid = d1.shopid.unique()
userid_shopid = []
for shopid in unique_shopid:
    userid = "0"
    if shopid in shop_list:
        max_value, max_user = shop_list[shopid]
        userid = "&".join([str(u) for u in sorted(list(max_user))])
    userid_shopid.append(userid)
sms = pd.DataFrame({
    "shopid": unique_shopid,
    "userid": userid_shopid
})
sms[sms.userid != "0"]
sms.to_csv("/kaggle/working/submission.csv", index=False)