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
# Load the file

order = pd.DataFrame(pd.read_csv(os.path.join(dirname, filename)))

order.head()



# Number of orders

num_order = order.shape[0]



# Number of unique sellers

num_seller = len(order.shopid.unique())



print('Number of orders:', num_order)

print('Number of unique shops:', num_seller)
# Groupby the shopid

pd_order = order.groupby('shopid')['orderid'].apply(list).to_frame()

pd_order['userid'] = order.groupby('shopid')['userid'].apply(list)

pd_order['event_time'] = order.groupby('shopid')['event_time'].apply(list)



pd_order['num_order'] = list(map(lambda x: len(x), pd_order.orderid))



# Count number of unique buyers per each shop

pd_order['num_unique_buyer'] = order.groupby('shopid')['userid'].nunique()



# Since order brushing only occurs if the number of orders per shop - number of unique buyers per shop is larger or equal to 2

pd_order = pd_order[pd_order.num_order - pd_order.num_unique_buyer >= 2]
# Function to identify the supicious buyer

def pick(alist):

    a = []

    for i in alist:

        if alist.count(i) >= 3:

            a.append(i)

    if a == []:

        return 0

    else:

        return sorted(list(set(a)))

    

pd_order['supicious_buyer'] = list(map(pick, pd_order.userid))
# Supicious order brushing

sup_buyer = pd_order[pd_order.supicious_buyer != 0]
dict_shop_user = dict(zip(pd_order.index, pd_order.supicious_buyer))
from datetime import datetime

timestamp = []

for t in list_time:

    timestamp.append(datetime.strptime(t,'%Y-%m-%d %H:%M:%S'))
def order_brushing_detection(shopid):

    users = dict_shop_user[shopid]

    fraud = []

    for user in users:

        list_time = sorted(list(np.array(pd_order.loc[shopid].event_time)[np.array(pd_order.loc[shopid].userid) == user]))

        timestamp = []

        for t in list_time:

            timestamp.append(datetime.strptime(t,'%Y-%m-%d %H:%M:%S'))

        diff_time = []

        for i in range(len(timestamp)-2):

            diff_time.append((timestamp[i+2] - timestamp[i]).seconds)

        diff_time = np.array(diff_time)

        if sum(diff_time < 3600):

            fraud.append(user)

    fraud = '&'.join([str(i) for i in sorted(fraud)])

    if fraud == '':

        return 0

    else:

        return fraud
fraud_buyer = list(map(order_brushing_detection, sup_buyer.index))
fraud_shop = pd.DataFrame(sup_buyer.index)
fraud_shop['userid'] = fraud_buyer
fraud_shop.index = fraud_shop.shopid
pd_order.supicious_buyer.to_frame()
fraud_shop.index = range(fraud_shop.shape[0])
fraud_shop
pd_order = order.groupby('shopid')['orderid'].apply(list).to_frame()

pd_order['userid'] = order.groupby('shopid')['userid'].apply(list)

pd_order['event_time'] = order.groupby('shopid')['event_time'].apply(list)



pd_order['num_order'] = list(map(lambda x: len(x), pd_order.orderid))



# Count number of unique buyers per each shop

pd_order['num_unique_buyer'] = order.groupby('shopid')['userid'].nunique()
pd_order = pd_order.userid.to_frame()
pd_order.userid = 0
pd_order.loc[fraud_shop.shopid]
fraud_shop.index = fraud_shop.shopid
for shop in fraud_shop.shopid:

    pd_order.loc[shop] = fraud_shop.loc[shop]
pd_order.to_csv('aaa.csv')