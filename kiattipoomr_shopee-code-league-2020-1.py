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
root = '/kaggle/input/students-order-brushing-1/'
df = pd.read_csv(root+'order_brush_order.csv')
df_c_order = df.groupby('shopid').count()
# shop with more than 3 buyers overall
df[~df['shopid'].isin(df_c_order[df_c_order['orderid']<3].index)].sort_values(by='shopid')
df.sort_values(by='event_time')[['shopid','userid','event_time']].values.tolist()
from datetime import *

checklist, contribute_count, record = dict(), dict(), dict()
for shop, user, time in df.sort_values(by='event_time')[['shopid','userid','event_time']].values.tolist() :
    if shop not in checklist :
        checklist[shop] = []
        contribute_count[shop] = dict()
    record[shop] = '0'
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    checklist[shop].append((user,time))
    if user not in contribute_count[shop] :
        contribute_count[shop][user] = 0
    contribute_count[shop][user] += 1

def printlist(shop):
    for order in checklist[shop] :
        print(order)

def find_max_contribute(shop) :
    user_list = []
    for user in sorted(contribute_count[shop].keys()) :
        if contribute_count[shop][user] == max(contribute_count[shop].values()) :
            user_list.append(str(user))
    return '&'.join(user_list)
print('Success')
for shop in sorted(checklist.keys()) :
    if len(checklist[shop]) > 2 :
        start, end = 0, 0
        unique_customers = [checklist[shop][0][0]]
        while True :
            #print(unique_customers)
            end += 1
            if end == len(checklist[shop]) : break
            else :
                #print(unique_customers)
                if checklist[shop][end][0] not in unique_customers :
                    unique_customers.append(checklist[shop][end][0])
                for i in range(start,end):
                    time_diff = checklist[shop][end][1] - checklist[shop][i][1]
                    if time_diff > timedelta(hours=1) :
                        if checklist[shop][i+1][0] != checklist[shop][i][0] and len(unique_customers) > 0:
                            unique_customers.pop(0)
                    else :
                        start = i
                        break 
                if not changed : start = end - 1
                if len(unique_customers) > 0 and (end - start) // len(unique_customers) >= 3 :
                    record[shop] = find_max_contribute(shop)
                    #print("Shop :",shop,'Substituted')
                    break
            
    #print('Shop',shop,"Done.")
print('Success')
X = []
for shopid in (record.keys()) :
    X.append([shopid,record[shopid]])

result = pd.DataFrame(X, columns=['shopid','userid'])

result.to_csv('submission.csv',header=True,index=False)