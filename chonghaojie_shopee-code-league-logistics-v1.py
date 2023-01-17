import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
data = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")
data.head()
time = datetime.fromtimestamp(1583138397)
print(time)
no_rows = data.count
print(no_rows)
data = data.fillna(0)
data['2nd_deliver_attempt'][0]
new_pick = []
new_1 = []
new_2 = []

for i in range(3176313):
    time_pick = datetime.fromtimestamp(data['pick'][i])
    time_1 = datetime.fromtimestamp(data['1st_deliver_attempt'][i])
    
    if data['2nd_deliver_attempt'][i] != 0:
        time_2 = datetime.fromtimestamp(data['2nd_deliver_attempt'][i])
        new_2.append(time_2)
    else:
        new_2.append(0)
    
    new_pick.append(time_pick)
    new_1.append(time_1)
  

a =new_1[0] - new_pick[0]
print(a)
x = a.days
new_1[10].day
df = pd.read_csv("../input/shopee-code-league/working_days.csv")
df.head()
df['working days'][0].dtype

is_late = []
for i in range(3176313):
    x1 = new_1[i] - new_pick[i]
    x = x1.days
    
    test = new_pick[i]
    
    for j in range(x):
        test = test +  timedelta(days=1)  
        if test.day == 8:
            x = x - 1
        elif test.day == 25:
            x = x - 1
        elif test.day == 30:
            x = x - 1
        elif test.day == 31:
            x = x - 1


    if new_2[i] != 0:
        y1 = new_2[i] - new_1[i]
        y = y1.days
        
        test2 = new_pick[i]
    
        for h in range(y):
            test2 = test +  timedelta(days=1)  
            if test2.day == 8:
                y = y - 1
            elif test2.day == 25:
                y = y - 1
            elif test2.day == 30:
                y = y - 1
            elif test2.day == 31:
                y = y - 1
    else:
        y = 0
    
    
    if x < df['working days'][i] and y < 3:
        is_late.append(0)
    else:
        is_late.append(1)    
data = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")
orderid = []

for i in range(3176313):
    orderid.append(data['orderid'][i])
results=pd.DataFrame({"orderid":orderid,
                      "is_late":is_late})

results.to_csv("results.csv",index=False)
