import numpy as np
import pandas as pd
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
# Any results you write to the current directory are saved as output.
df = DataFrame({'类别':['水果','水果','水果','蔬菜','蔬菜','肉类','肉类'],
                '产地':['美国','中国','中国','中国','新西兰','新西兰','美国'],
                '水果':['苹果','梨','草莓','番茄','黄瓜','羊肉','牛肉'],
               '数量':[5,5,9,3,2,10,8],
               '价格':[5,5,10,3,3,13,20]})
print(df)
print(pd.crosstab([df['类别'],df['产地']],df['水果'],margins=True)) # 按类别分组，统计各个分组中产地的频数
import datetime
FEATURE_EXTRACTION_SLOT  = 10
day = datetime.timedelta(days=FEATURE_EXTRACTION_SLOT+2)
LabelDay = datetime.datetime(2014,12,18,0,0,0)
print(LabelDay-day)
print(day)
user_table = pd.read_csv('../input/dataset/tianchi_fresh_comp_train_user.csv')
item_table = pd.read_csv('../input/dataset/tianchi_fresh_comp_train_item.csv')
user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])
user_table = user_table[user_table['days'] != '2014-12-12']
user_table = user_table[user_table['days'] != '2014-12-11']
user_table.head()
#user_table.to_csv('../input/drop1112_sub_item.csv',index=None)
LabelDay = datetime.datetime(2014,12,18,0,0,0)
user_table.head()
user_table.to_csv('drop1112_sub_item.csv',index=None)