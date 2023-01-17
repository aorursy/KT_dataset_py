# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



import pandas as pd
from pandas import Series, DataFrame # pandas中两个最常用的数据类型
obj = pd.Series([1,2,3])
obj
obj.index
obj.values
# 自己给标定index的label,当然默认的数字index还是可以用的，只不过我们自己又加了个别名
obj2 = pd.Series([4,7,-5,3], index=['d','b','c','a'])
obj2
obj2[1]
obj2.index
obj2.values
obj2['a']
temp = obj2[['c','b','a']] # 返回的仍然是Series
type(temp)
obj2[obj2 > 0]
obj2 * 2
import numpy as np
np.exp(obj2)
'b' in obj2 # 这是判读索引是否在的方法,等同于obj2默认表示obj2.index了嘛
7 in obj2 # 判断是否
'b' in obj2.index
7 in obj2.values
type(obj2.index)
type(obj2.values)
type(obj2)
sdata = {'a': 1, 'b':2, 'c':3}
obj3 = pd.Series(sdata)
obj3
states = ['How','are','you']
obj4 = pd.Series(sdata,index=states)
obj4
new_states = ['a','b','Hello']
obj5 = pd.Series(sdata,index=new_states)
obj5
pd.isnull(obj4)
pd.isnull(obj5)
temp = pd.notnull(obj5)
type(temp) # 生成的是Series
obj3 + obj5
obj3
obj5
obj5.name # 默认空值
obj5.name = "test"
obj5.index.name = "temp"
obj5
# 这是直接修改index的内容
obj.index
obj.index = ["How","Are","You"]
obj
