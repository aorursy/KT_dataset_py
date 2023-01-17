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
df_order = pd.read_csv('../input/open-shopee-code-league-logistic/delivery_orders_march.csv')
df_SLA = pd.read_excel('../input/open-shopee-code-league-logistic/SLA_matrix.xlsx')
df_order.head()
df_SLA.head()
SLA_Matrix  = [[3, 5, 7, 7], [5, 5, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]]
origin = []
for i in df_order.buyeraddress.tolist() :
    if 'metro manila' in i.lower():
        origin.append(1)
    elif 'luzon' in i.lower():
        origin.append(2)
    elif 'visayas' in i.lower():
        origin.append(3)
    elif 'mindanao' in i.lower():
        origin.append(4)

destination = []
for i in df_order.selleraddress.tolist() :
    if 'metro manila' in i.lower():
        destination.append(1)
    elif 'luzon' in i.lower():
        destination.append(2)
    elif 'visayas' in i.lower():
        destination.append(3)
    elif 'mindanao' in i.lower():
        destination.append(4)

time_needed = []
for (o, d) in zip(origin, destination):
    time_needed.append(SLA_Matrix[o-1][d-1])

print(time_needed.count(3))
print(time_needed.count(5))
print(time_needed.count(7))
import datetime

def converttodate(epoch_time):
    return datetime.datetime.fromtimestamp(epoch_time).strftime("%Y-%m-%d")
pick_date = df_order.pick.tolist()
first_attempt = df_order['1st_deliver_attempt'].tolist()
second_attempt = df_order['2nd_deliver_attempt'].tolist()

pick = []
first = []
second = []

for (i, j, k) in zip(pick_date, first_attempt, second_attempt):
    pick.append(converttodate(i))
    first.append(converttodate(j))
    if np.isnan(k) :
        second.append(np.nan)
    else :
        second.append(converttodate(k))
df_solution = pd.DataFrame();
df_solution['orderid'] = df_order.orderid
df_solution['origin'] = origin
df_solution['destination'] = destination
df_solution['time_needed'] = time_needed
df_solution['pick'] = pick
df_solution['first'] = first
df_solution['second'] = second

df_solution.dtypes
is_late = []
weekmask = [1, 1, 1, 1, 1, 1, 0]
for (i, j, k, l) in zip(df_solution['pick'].tolist(), df_solution['first'].tolist(), df_solution['second'].tolist() ,df_solution['time_needed']):
    if (abs(np.busday_count(j,i)) - 1 > l) :
        is_late.append(1)
    else :
        is_late.append(0)
df_submit = pd.DataFrame();
df_submit['orderid'] = df_solution['orderid']
df_submit['is_late'] = is_late

df_submit.shape
df_submit.to_csv('solution_noholiday.csv',index=False)
