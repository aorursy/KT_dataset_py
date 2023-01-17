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
march_orders_path = '/kaggle/input/logistics-shopee-code-league/delivery_orders_march.csv'

sla_matrix = '/kaggle/input/logistics-shopee-code-league/SLA_matrix.xlsx'

orders_df = pd.read_csv(march_orders_path)

matrix_df = pd.read_excel(sla_matrix)
orders_df.head()
orders_df.dtypes
orders_df['pick'] = pd.to_datetime(orders_df['pick'])

orders_df['1st_deliver_attempt'] = pd.to_datetime(orders_df['1st_deliver_attempt'])

orders_df.head(30)
import re



test = orders_df.loc[orders_df['buyeraddress'].str.contains(pat = 'visayas|mindanao',case = False, flags = re.I, regex=True)]

test.head(20)

sla_df = orders_df.copy()
sla_df.head()
sla_df.drop(columns = ['pick', '1st_deliver_attempt','2nd_deliver_attempt'])
sla_df['buyer_seller_add'] = sla_df['buyeraddress'] + sla_df['selleraddress']

sla_df = sla_df.drop(columns = ['buyeraddress', 'selleraddress','1st_deliver_attempt','2nd_deliver_attempt'])
sla_df['SLA_7'] = sla_df['buyer_seller_add'].str.contains(pat = 'visayas|mindanao',case = False, flags = re.I, regex=True)
sla_df['SLA_5'] = sla_df['buyer_seller_add'].str.contains(pat = 'luzon',case = False, flags = re.I, regex=True)
sla_df['SLA_3'] = sla_df['buyer_seller_add'].str.contains(pat = 'manila',case = False, flags = re.I, regex=True)
sla_df.head(30)
sla_df['SLA'] = 0

if ([sla_df['SLA_3'] == True]):

    sla_df['SLA'].loc[sla_df['SLA_3'] == True] = 3

    

if ([sla_df['SLA_5'] == True]):

    sla_df['SLA'].loc[sla_df['SLA_5'] == True] = 5

    

if ([sla_df['SLA_7'] == True]):

    sla_df['SLA'].loc[sla_df['SLA_7'] == True] = 7



sla_df.head(30)
sampledf = pd.read_csv(march_orders_path)

sla_df['first'] = sampledf['1st_deliver_attempt']

sla_df['second'] = sampledf['2nd_deliver_attempt']
sla_df.head()
sla_df['pick'] = sampledf['pick']

sla_df['first_diff'] = sla_df['first'] - sla_df['pick']

sla_df['second_diff'] = sla_df['second'] - sla_df['pick']

sla_df['final_diff'] = sla_df.apply(lambda row: row['first_diff'] if np.isnan(row['second_diff']) else row['second_diff'], axis=1)
TOTAL_SEC = 60 * 60 * 24

sla_df['days'] = sla_df['final_diff'] / TOTAL_SEC

sla_df['is_late'] = sla_df.apply(lambda row: 1 if (row['days'] > row['SLA']) else 0, axis=1)

sla_df.head()
ans_df = sla_df[['orderid', 'is_late']]

ans_df.to_csv('/kaggle/working/results.csv', index=False)
