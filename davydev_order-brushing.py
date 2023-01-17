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
import numpy as np # linear algebra
import pandas as pd # data processing

# import `order brushing` dataset
datadir = '/kaggle/input/shopee-code-league-20/_DA_Order_Brushing/order_brush_order.csv'
df = pd.read_csv(datadir)
from datetime import datetime, timedelta
from collections import defaultdict

shop = defaultdict(list)

for i in range(df.shape[0]):
    oid, sid, uid, et = df.iloc[i]
    et = datetime.strptime(et, '%Y-%m-%d %H:%M:%S')
    shop[sid].append((et, uid, oid))
one_hour = timedelta(hours=1.)
two_hour = timedelta(hours=2.)

def detect_segment(events, left, right):
    assert(left <= right and left >=0 and right < len(events))
    dic = defaultdict(list)
    for i in range(left, right+1):
        dic[events[i][1]].append(events[i][2])
    
    brushing = (right-left+1) >= 3*len(dic)
    
    return brushing, dic
    

def detect(events):
    
    # [left, right]
    
    ## order id => user id
    ### consider orders in important []
    ooo = {}
    ff=False
    
    
    start = events[ 0][0]
    end   = events[-1][0]
    
    events.insert(0, (start - two_hour, ))
    events.append(   ( end  + two_hour, ))
    
    for left in range(1, len(events)-1):
        for right in range(left, len(events)-1):
            if events[right][0] - events[left][0] <= one_hour:
                if (events[right+1][0] - events[left-1][0] > one_hour and 
                    events[right+1][0] != events[right][0] and 
                    events[left][0] != events[left-1][0]):
                    brushing, dic = detect_segment(events, left, right)
                    if brushing:
                        ff=True
                        for (uid, loid) in dic.items():
                            for order in loid:
                                ooo[order] = uid

    ## 1 2
    ## 1:30 - 2:20
    ## 
    ## ooo => order IDs in ... 
    
    ## uu : uid => count of # orders
    uuu = defaultdict(int)
    
    ## mm: maximum number of orders 
    mm = -1
    for oid, uid in ooo.items():
        uuu[uid] += 1
        mm = max(mm, uuu[uid])
    
    ## candidates
    can = []
    
    for uid, count in uuu.items():
        if count == mm:
            can.append(uid)
    
    if len(can) == 0:
        return ff, '0'
    else:
        can.sort()
        return ff, '&'.join([str(a) for a in can])
data = []
c=0
for sid, v in shop.items():
    v.sort()
    ff, out = detect(v)
    if ff:
        c=c+1
        print(c)
    data.append((sid, out))


submission = pd.DataFrame(data=data, columns = ['shopid', 'userid'])
submission.to_csv('submission_order_brushing.csv', index=False, header=True)
submission.shape[0] - sum(submission['userid'] == '0')
!head submission_order_brushing.csv