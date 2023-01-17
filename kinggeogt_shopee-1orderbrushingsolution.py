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
brush = pd.read_csv("/kaggle/input/order_brush_order.csv")
brush
brush.isnull().info()
print(type(brush.event_time[0]))
print(brush.dtypes)
print(brush.size, brush.shape)
# print(brush.event_time[0])
brush.shopid.unique
brush.userid.unique
brush.insert(3,"counter",0)
brush
brush.insert(4,"brushing", 0)
brush
brush.insert(5,"event_split",0)
brush
brush.loc[(brush.shopid==164933170)]
brush.event_time[0][:10]
done = []

for i in range(brush.shape[0]):
    orderid = brush.shopid[i]
    if orderid in done:
        continue
    done.append(orderid)
    
    
len(done)
done
output = pd.DataFrame({'shopid':done, 'userid':brush.counter[:18770]})
output.to_csv('submit1.csv', index=False)
ts = brush.loc[(brush.shopid == done[0])]
ts
ts.index
from datetime import time, timedelta
old = 0
new = 0
listindex = ts.index
buyer = []
order = []

for j in range(1,len(listindex)):
    old = ts.event_time[listindex[j-1]]
    olddate = old[:10]
    old = old[11:19]
    new = ts.event_time[listindex[j]]
    newdate = new[:10]
    new = new[11:19]
    diff = int(new[0:2]) - int(old[0:2])
    if diff<=1 and olddate == newdate:
        if not ts.userid[listindex[j]] in order:
            buyer.append(ts.userid[listindex[j]])
        order.append(ts.orderid[listindex[j]])
        print(buyer)
        print("ini", order)
    else:
        print(olddate, newdate)
    
output2 = pd.DataFrame({'shopid':done, 'userid':brush.counter[:18770], 'counter':brush.counter[:18770]},index=done)
output2.counter[93950878] = 1
print(output2)
from datetime import time, timedelta
old = 0
new = 0


for i in done:
    ts = brush.loc[(brush.shopid == i)]
    listindex = ts.index
    buyer = []
    order = []
    
    for j in range(1,len(listindex)):
        old = ts.event_time[listindex[j-1]]
        olddate = old[:10]
        old = old[11:19]
        new = ts.event_time[listindex[j]]
        newdate = new[:10]
        new = new[11:19]
        diff = int(new[0:2]) - int(old[0:2])
        if diff<=1 and olddate == newdate:
            if not ts.userid[listindex[j]] in order:
                buyer.append(ts.userid[listindex[j]])
            order.append(ts.orderid[listindex[j]])
        else:
            continue
    print(buyer, order)
    if len(order) > 0:
        counter = len(buyer)/len(order)
    else:
        counter = 0
#     if counter >= 3:
#         buyer.sort()
#         output2.userid[i] = buyer[0]
#     else:
#         output2.userid[i] = 0
    if counter > 0:
        output2.userid[i] = buyer[0]
    else:
        output2.userid[i] = 0
output2.drop(['counter'], axis=1)
output2.to_csv('submit2.csv', index=False)
output2