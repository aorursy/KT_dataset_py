# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from pandas import DataFrame,Series
df1 = DataFrame({'key':['A','B','C','D','E'],'data1':[10,33,12,42,23]})
df1
df2 = DataFrame({'key':['A','Q','E'],'data2':[70,366,23]})
df2
#merge similar rows
pd.merge(df1,df2)
#key is the colomn that is compared for merging
#how specify whether left/right join
pd.merge(df1,df2,on='key',how='left')
#concatunation
pd.concat([df1,df1])
pd.concat([df1,df1],axis=1)
ser1 = Series([1,2,3,4])
ser2 = Series([5,3,6,5])

pd.concat([ser1,ser2])
pd.concat([ser1,ser2],axis=1)
df_new = DataFrame({'key':['A','A','A','B','B'],'data':[2,3,2,3,2]})
df_new
df_new.duplicated()
df_new.drop_duplicates()
#Drop the duplicates with key colomns
df_new.drop_duplicates(['key'])
#Map a colomn with another set and create a new colomn in dataframe
df_state = DataFrame({'State':['Kerala','Tamil','Karnataka']})
capital_map = {'Kerala':'TVM','Tamil':'Chennai','Karnataka':'Bangalore'}

df_state['capital'] = df_state['State'].map(capital_map)
df_state
ser12 = Series([1,3,4,6,2,5,2,6])
ser12.replace([1,5],[6,np.nan])
