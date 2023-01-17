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
def csv_reader(filename : str):
    return pd.read_csv(filename)

def convert_to_dict(data_frame, key_column : str, *value_column):
    return data_frame.groupby(key_column)[value_column].apply(lambda f: f.values.tolist()).to_dict()

def check(order_count, user_count):
    return True if (order_count / user_count) >= 3 else False
file_path = "/kaggle/input/order_brush_order.csv"
home_data = pd.read_csv(file_path)
home_data['event_time'] = pd.to_datetime(home_data['event_time'], format='%Y-%m-%d %H:%M:%S')
home_data = home_data.sort_values(by=['shopid', 'event_time', 'userid'])
order = convert_to_dict(home_data, 'shopid', 'event_time', 'userid')
shop_id = []
user_id = []
for shop in order:
    shop_id.append(shop)
    transaction = order[shop]
    suspicious = []
    for i in range(len(order[shop])):
        count = 0
        suspect = []
        current_transaction = transaction[i]
        for j in range(i, len(order[shop])):
            next_transaction = transaction[j]
            if (next_transaction[0] < (current_transaction[0] + datetime.timedelta(hours=1))):
                count += 1
                if next_transaction[1] not in suspect:
                    suspect.append(next_transaction[1])
        if check(count, len(suspect)) == True:
            for target in suspect:
                if target not in suspicious:
                    suspicious.append(target)
    if len(suspicious) > 0:
        suspicious.sort()
        temp = ''
        for a in range(len(suspicious)):
            if a > 0:
                temp += '&'
            temp += str(suspicious[a])
        user_id.append(temp)
    else:
        user_id.append('0')
result = pd.DataFrame({'shopid': shop_id, 'userid': user_id})
result
result.to_csv('result(123).csv', index=False)