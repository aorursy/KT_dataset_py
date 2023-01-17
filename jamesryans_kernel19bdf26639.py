# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = "../input/students-order-brushing-1/order_brush_order.csv"
home_data = pd.read_csv(file_path)
home_data.sort_values(by = ['shopid', 'event_time'], ascending = [True, True], inplace=True)
print(home_data.head(20))
suspicious = {}

shop = -1

current_user = {}
time_list = []

def calculate():
    if (shop == -1):
        return
    
    suspicious[shop] = {0:-1}
    
    if (shop == 10084):
        print(time_list)
    
    r = -1
    for l in range(len(time_list)):
        while (r + 1 < len(time_list) and int((time_list[r + 1][1] - time_list[l][1]).total_seconds()) <= 3600):
            r += 1

            if (time_list[r][0] not in current_user):
                current_user[time_list[r][0]] = 0

            current_user[time_list[r][0]] += 1
            
        if (shop == 10084):
            print(l, r, current_user)
            
        if (((r - l + 1) // len(current_user)) >= 3):
            most = 0
            user = -1

            for key, value in current_user.items():
                if (value > most):
                    user, most = key, value

            if (user not in suspicious[shop]):
                suspicious[shop][user] = 0

            suspicious[shop][user] = max(suspicious[shop][user], most)
            
        current_user[time_list[l][0]] -= 1
        if (current_user[time_list[l][0]] == 0):
            del current_user[time_list[l][0]]

for id, row in home_data.iterrows():
    if (row['shopid'] != shop):
        calculate()
        
        shop = row['shopid']
        current_user = {}
        time_list = []
        
    time_list.append((row['userid'], datetime.datetime.strptime(row['event_time'], '%Y-%m-%d %H:%M:%S')))
    
calculate()
print(suspicious[10009])
print(suspicious[10051])
print(suspicious[10061])
print(suspicious[10084])
shop_id = []
user_id = []

for nama, users in suspicious.items():
    kumpulan = [(value, key) for key, value in users.items()]
    best = max(kumpulan)[0]
    
#     if (nama == 10084):
#         print(kumpulan)
    
    shop_id.append(str(nama))
    
    tmp = []
    
    for value, key in kumpulan:
        if (value == best):
            tmp.append(key)
            
    tmp.sort()
    
#     user_id.append('0')
    user_id.append("&".join(str(x) for x in tmp))
    
hasil = {'shopid' : shop_id,
        'userid' : user_id}

print(hasil)

df = pd.DataFrame(hasil, columns=['shopid', 'userid'])

print(df)

print(df.loc[1, 'shopid'])
print(type(df.loc[1, 'userid']))

df.to_csv('Solution.csv', index = False, header=True, na_rep = '0')
    
# print(shop_id)
# print(user_id)
help(df.to_csv)
home_data.to_excel('temp.xlsx', header = True)