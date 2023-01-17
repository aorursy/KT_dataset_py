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
os.getcwd()
root_path = os.path.join('/kaggle/input',"shopee-code-league-20/_DA_Logistics")
os.path.getsize(os.path.join(root_path, "delivery_orders_march.csv"))
order_list = pd.read_csv(os.path.join(root_path, "delivery_orders_march.csv"))
order_list.shape
pd.set_option('display.max_colwidth', None)

order_list.head()
order_list["buyer_metro"] = order_list.buyeraddress.str.split().str[-1].str.lower()

order_list["seller_metro"] = order_list.selleraddress.str.split().str[-1].str.lower()
order_list["buyer_metro"].str.lower().value_counts()
order_list["buyer_metro"].str.lower().value_counts().sum()
order_list["seller_metro"].value_counts()
order_list["seller_metro"].value_counts().sum()
del order_list["buyeraddress"]

del order_list["selleraddress"]

order_list.head()
order_list["pick"] = pd.to_datetime(order_list.pick,unit='s').dt.tz_localize('utc').dt.tz_convert('Asia/Manila')

order_list["1st_deliver_attempt"] = pd.to_datetime(order_list["1st_deliver_attempt"],unit='s').dt.tz_localize('utc').dt.tz_convert('Asia/Manila')

order_list["2nd_deliver_attempt"] = pd.to_datetime(order_list["2nd_deliver_attempt"],unit='s').dt.tz_localize('utc').dt.tz_convert('Asia/Manila')
order_list['pick'] = order_list['pick'].values.astype('datetime64[D]')

order_list["1st_deliver_attempt"] = order_list["1st_deliver_attempt"].values.astype('datetime64[D]')

order_list["2nd_deliver_attempt"] = order_list["2nd_deliver_attempt"].values.astype('datetime64[D]')

order_list.head()
holidays = ["2020-03-08", "2020-03-25" , "2020-03-30", "2020-03-31"]

order_list["day_to_1st_attempt"] = np.busday_count(order_list['pick'].values.astype('datetime64[D]'), order_list["1st_deliver_attempt"].values.astype('datetime64[D]'), weekmask='1111110',holidays=holidays)



f = order_list.dropna().copy()

f["day_to_2nd_attempt"] = np.busday_count(f["1st_deliver_attempt"].values.astype('datetime64[D]'), f["2nd_deliver_attempt"].values.astype('datetime64[D]'), weekmask='1111110')  



order_list = pd.merge(order_list,f[["day_to_2nd_attempt"]],how='left',left_index=True,right_index=True)

order_list.head()
days = []

for i, j in order_list[['buyer_metro','seller_metro']].itertuples(index=False):

        if i == 'manila'and j == 'manila':

            days.append(3)

        elif (i == 'manila' and j == 'luzon') or (i == 'luzon' and (j == 'manila' or j == 'luzon')):

            days.append(5)

        else:

            days.append(7)



order_list['days_limit'] = days

order_list.head()
late_flag = []

for i, j, k in order_list[['day_to_1st_attempt','day_to_2nd_attempt','days_limit']].itertuples(index=False):

        if i > k:

            late_flag.append(1)

        elif j > 3:

            late_flag.append(1)

        else:

            late_flag.append(0)

order_list["late"] = late_flag
order_list["late"].value_counts()
order_list["late"].value_counts().plot.pie(figsize=(10,8),autopct='%1.2f%%')

submission = pd.DataFrame({'orderid':order_list['orderid'], 'is_late':order_list['late'].apply(int)})



submission
submission.is_late.value_counts()