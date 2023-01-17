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
delivery_orders = pd.read_csv('/kaggle/input/logistics-shopee-code-league/delivery_orders_march.csv')
hols = ['2020-03-08','2020-03-25','2020-03-30','2020-03-31']
SLA = {

    'manila_manila': 3,

    'manila_luzon': 5,

    'manila_visayas': 7,

    'manila_mindanao': 7,

    'luzon_manila':5,

    'luzon_luzon':5,

    'luzon_visayas': 7,

    'luzon_mindanao': 7,

    'visayas_manila':7,

    'visayas_luzon':7,

    'visayas_visayas': 7,

    'visayas_mindanao': 7,

    'mindanao_manila': 7,

    'mindanao_luzon': 7,

    'mindanao_visayas': 7,

    'mindanao_mindanao': 7,

}
delivery_orders['origin'] = [buyeraddress.split()[-1].lower() for buyeraddress in delivery_orders.buyeraddress]

delivery_orders['destination'] = [selleraddress.split()[-1].lower() for selleraddress in delivery_orders.selleraddress]

delivery_orders = delivery_orders.drop(columns=['buyeraddress', 'selleraddress'])
delivery_orders[['pick','1st_deliver_attempt','2nd_deliver_attempt']] += 8*60*60
delivery_orders['pick'] = pd.to_datetime(delivery_orders['pick'], unit='s').dt.date

delivery_orders['1st_deliver_attempt'] = pd.to_datetime(delivery_orders['1st_deliver_attempt'], unit='s').dt.date

delivery_orders['2nd_deliver_attempt'] = pd.to_datetime(delivery_orders['2nd_deliver_attempt'], unit='s').dt.date
delivery_orders.loc[delivery_orders['2nd_deliver_attempt'].isna(), '2nd_deliver_attempt'] = 0

delivery_orders['1st_deliver_attempt_gap'] = np.busday_count(delivery_orders['pick'], delivery_orders['1st_deliver_attempt'],weekmask='1111110',holidays=hols)

delivery_orders['2nd_deliver_attempt_gap'] = np.busday_count(delivery_orders['1st_deliver_attempt'], delivery_orders['2nd_deliver_attempt'],weekmask='1111110',holidays=hols)

delivery_orders.loc[delivery_orders['2nd_deliver_attempt_gap'] < 0, '2nd_deliver_attempt_gap'] = 0
delivery_orders['SLA'] = 0
for origin in delivery_orders.origin.unique():

    for destination in delivery_orders.destination.unique():

        mask = (delivery_orders['origin'] == origin) & (delivery_orders['destination'] == destination)

        delivery_orders.loc[mask, 'SLA'] = SLA[origin+'_'+destination]
delivery_orders['is_late'] = 0
mask = (delivery_orders['1st_deliver_attempt_gap'] > delivery_orders['SLA']) | (delivery_orders['2nd_deliver_attempt_gap'] > 3)

delivery_orders.loc[mask, 'is_late'] = 1
submissions_df = delivery_orders[['orderid', 'is_late']].copy()
submissions_df.to_csv('submission.csv', index=False)