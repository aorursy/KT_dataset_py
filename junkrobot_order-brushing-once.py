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
df = pd.read_csv('../input/order-brushing-dataset-shopee-code-league-week-1/order_brush_order.csv')
df.head()
df.info()
# change event_time datatype to datetime

df.event_time = pd.to_datetime(df.event_time)

df.info()
df.groupby('shopid').orderid.size().describe()
unique_shop = df.shopid.drop_duplicates()

unique_shop.size
submission = pd.DataFrame(unique_shop.copy())

submission['userid'] = ''

submission.head()
for shop in unique_shop:

    shop_orders = (df[df.shopid == shop].sort_values(by='event_time'))

    brushser = set()

    max_value = 0

    max_shop = ''

    for index, order in shop_orders.iterrows():

        order_by_hours = shop_orders[(shop_orders.event_time >= order.event_time) & (shop_orders.event_time <= (order.event_time + pd.Timedelta('1 hours')))]

        # ignore if order less than 3

        if order_by_hours.orderid.size < 3:

            continue

        rating = order_by_hours.orderid.count() / order_by_hours.userid.unique().size

        # ignore if order have rating less than 3

        if rating < 3:

            continue

        suspicious_buyer = order_by_hours.userid.value_counts()

        suspicious_buyer = suspicious_buyer[suspicious_buyer == suspicious_buyer.max()] 

        for index, value in suspicious_buyer.items():

            if value > max_value:

                max_value = value

                if len(brushser) > 0:

                    brushser.pop()

                brushser.add(index)

            elif value == max_value:

                brushser.add(index)

    if len(brushser) > 0:

        submission.loc[submission.shopid == shop,'userid'] = '&'.join(map(str, brushser)) # this for multiple crushing users

    else:

        submission.loc[submission.shopid == shop,'userid'] = '0' # no crushing order
submission.to_csv('submission.csv', index=False)